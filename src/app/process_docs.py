import os
import fitz  # PyMuPDF for PDF handling
import pandas as pd
import logging
from typing import List, Dict, Any
from utils.config_utils import load_config
from app.embeddings.openai_embeddings import OpenAIEmbeddingGenerator
from app.embeddings.vertex_embeddings import VertexEmbeddingGenerator

# Load configuration
config = load_config()

if config is None:
    logging.error("Configuration could not be loaded. Please check the config.yaml file.")
    raise SystemExit("Exiting due to missing configuration.")

# Function to select embedding generator
def get_embedding_generator(model: str):
    logging.debug(f"Selecting embedding generator for model: {model}")
    if model == "openai":
        openai_config = config['embeddings']['models']['openai']
        return OpenAIEmbeddingGenerator(api_key=openai_config['api_key'], model=openai_config['model'])
    elif model == "vertex":
        vertex_config = config['embeddings']['models']['vertex']
        return VertexEmbeddingGenerator(project_id=vertex_config['project_id'], location=vertex_config['location'])
    else:
        raise ValueError(f"Unsupported embedding model: {model}")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    logging.info(f"Extracting text from {pdf_path}")
    pdf_document = fitz.open(pdf_path)
    text_data = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        text_data.append({'file_name': os.path.basename(pdf_path), 'page_num': page_num + 1, 'text': text})

    logging.info(f"Extracted {len(text_data)} pages from {pdf_path}")
    return text_data

# Function to generate text chunks
def generate_chunks(text: str, chunk_size: int = None) -> List[str]:
    if chunk_size is None:
        chunk_size = config['embeddings']['default_chunk_size']
    logging.debug(f"Generating chunks of size {chunk_size}")
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    logging.info(f"Generated {len(chunks)} chunks of text")
    return chunks

# Function to process a PDF and return a DataFrame with text and embeddings
def process_pdf(pdf_path: str, chunk_size: int = None, model: str = None) -> pd.DataFrame:
    if model is None:
        model = config['embeddings']['default']
    if chunk_size is None:
        chunk_size = config['embeddings']['models'][model]['chunk_size']

    logging.info(f"Processing PDF: {pdf_path} with model: {model} and chunk_size: {chunk_size}")
    embedding_generator = get_embedding_generator(model)
    text_data = extract_text_from_pdf(pdf_path)
    all_chunks = []

    for page in text_data:
        chunks = generate_chunks(page['text'], chunk_size=chunk_size)
        for chunk_num, chunk_text in enumerate(chunks):
            logging.debug(f"Generating embedding for chunk {chunk_num + 1} on page {page['page_num']}")
            embedding = embedding_generator.generate(chunk_text)
            all_chunks.append({
                'file_name': page['file_name'],
                'page_num': page['page_num'],
                'chunk_number': chunk_num + 1,
                'chunk_text': chunk_text,
                'embedding': embedding
            })

    logging.info(f"Processed {len(all_chunks)} chunks for PDF: {pdf_path}")
    df_chunks = pd.DataFrame(all_chunks)
    return df_chunks

# Function to save DataFrame to CSV with model suffix
def save_to_csv(df: pd.DataFrame, output_path: str, model: str):
    output_path = f"{os.path.splitext(output_path)[0]}_{model}.csv"
    logging.info(f"Saving processed data to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Data saved successfully to {output_path}")

# Main function to process documents
def process_docs(output_dir: str, input_dir: str = None, pdf_file: str = None, model: str = None):
    if model is None:
        model = config['embeddings']['default']

    if input_dir:
        logging.info(f"Starting document processing from directory {input_dir} to {output_dir} using {model} model")
        pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pdf')]
    elif pdf_file:
        logging.info(f"Starting document processing for file {pdf_file} to {output_dir} using {model} model")
        pdf_files = [pdf_file]
    else:
        raise ValueError("Either input_dir or pdf_file must be provided.")
    
    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            logging.warning(f"File {pdf_path} does not exist and will be skipped.")
            continue
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_processed.csv")
        logging.debug(f"Processing file {pdf_path} with output to {output_path}")
        df_chunks = process_pdf(pdf_path, model=model)
        save_to_csv(df_chunks, output_path, model)
    logging.info(f"Document processing completed for directory {input_dir} or file {pdf_file}")
