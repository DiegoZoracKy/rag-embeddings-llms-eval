import os
import logging
import pandas as pd
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from utils.hash_utils import generate_query_hash
from utils.config_utils import load_config
from app.embeddings.openai_embeddings import OpenAIEmbeddingGenerator
from app.embeddings.vertex_embeddings import VertexEmbeddingGenerator

# Load configuration
config = load_config()

if config is None:
    logging.error("Configuration could not be loaded. Please check the config.yaml file.")
    raise SystemExit("Exiting due to missing configuration.")

# Function to select the embedding generator based on the model
def get_embedding_generator(model: str):
    logging.debug(f"Selecting embedding generator for model: {model}")
    if model == "openai":
        openai_config = config['embeddings']['models']['openai']
        return OpenAIEmbeddingGenerator(api_key=openai_config['api_key'], model=openai_config['model'])
    elif model == "vertex":
        vertex_config = config['embeddings']['models']['vertex']
        return VertexEmbeddingGenerator(
            project_id=vertex_config['project_id'],
            location=vertex_config['location'],
            model_name=vertex_config['model']
        )
    else:
        raise ValueError(f"Unsupported embedding model: {model}")

# Function to process the query
def process_query(query: str, embeddings_csv: str, top_k: int, output_dir: str, model: str):
    logging.info(f"Processing query: {query} with model: {model}")
    query_hash = generate_query_hash(query)
    query_dir = os.path.join(output_dir, query_hash)
    os.makedirs(query_dir, exist_ok=True)
    logging.debug(f"Created query directory at {query_dir}")

    # Load the embeddings
    if not os.path.exists(embeddings_csv):
        logging.error(f"Embeddings file {embeddings_csv} does not exist.")
        raise FileNotFoundError(f"{embeddings_csv} not found.")
    logging.info(f"Loading embeddings from {embeddings_csv}")
    df_chunks = pd.read_csv(embeddings_csv)
    logging.debug(f"Loaded {len(df_chunks)} embeddings from {embeddings_csv}")

    # Generate query embedding
    embedding_generator = get_embedding_generator(model)
    query_embedding = embedding_generator.generate(query)
    logging.debug(f"Generated embedding for query: {query}")

    # Calculate similarities
    logging.info(f"Calculating similarities for top {top_k} chunks.")
    df_chunks['similarity'] = df_chunks['embedding'].apply(lambda x: cosine_similarity([query_embedding], [eval(x)])[0][0])
    top_chunks = df_chunks.nlargest(top_k, 'similarity')
    logging.debug(f"Selected top {top_k} chunks based on similarity.")

    # Create result DataFrame with selected columns and rank
    result_df = top_chunks[['file_name', 'page_num', 'chunk_number', 'chunk_text', 'similarity']].reset_index(drop=True)
    result_df['rank'] = result_df.index + 1  # Add rank based on the index
    logging.info(f"Created result DataFrame with ranks.")

    # Save the result DataFrame
    top_chunks_file = os.path.join(query_dir, f"top_chunks_{model}.csv")
    result_df.to_csv(top_chunks_file, index=False)
    logging.info(f"Top chunks saved to {top_chunks_file}")

    # Save the query for reference
    query_file = os.path.join(query_dir, "query.txt")
    with open(query_file, 'w') as f:
        f.write(query)
    logging.info(f"Query saved to {query_file}")

    return top_chunks_file
