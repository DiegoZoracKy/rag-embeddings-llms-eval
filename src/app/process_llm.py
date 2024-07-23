# src/app/process_llm.py

import os
import logging
import json
import pandas as pd
from typing import Any, List, Dict, Optional
from utils.hash_utils import generate_query_hash
from app.process_query import process_query
from utils.config_utils import load_config
from app.llm.anthropic_llm import AnthropicLLM
from app.llm.openai_llm import OpenAILLM
from app.llm.vertex_llm import VertexLLM

# Load configuration
config = load_config()

if config is None:
    logging.error("Configuration could not be loaded. Please check the config.yaml file.")
    raise SystemExit("Exiting due to missing configuration.")

# Function to prepare the LLM prompt
def prepare_llm_prompt(query: str, top_chunks: List[Dict], system_message: str) -> Dict[str, str]:
    context = "\n".join(chunk['chunk_text'] for chunk in top_chunks)
    prompt_text = f"Questão: {query}\nContexto: {context}"
    logging.info(f"Prepared LLM prompt for query: {query}")
    return {
        "query": query,
        "context": context,
        "system_message": system_message,
        "text": prompt_text
    }

# Function to save the prompt
def save_prompt(prompt_data: Dict[str, str], query_dir: str, model: str):
    prompt_file = os.path.join(query_dir, f"prompt_{model}.txt")
    os.makedirs(os.path.dirname(prompt_file), exist_ok=True)  # Ensure the directory exists
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt_data["text"])
    logging.info(f"Prompt saved to {prompt_file}")
    return prompt_file

# Function to save the LLM result
def save_llm_result(response: str, prompt_sent: str, query_dir: str, model: str):
    result_file = os.path.join(query_dir, f"llm_response_{model}.txt")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)  # Ensure the directory exists
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(response)
    logging.info(f"LLM response saved to {result_file}")

    # Save the exact prompt sent to the LLM
    prompt_file = os.path.join(query_dir, f"sent_prompt_{model}.txt")
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(str(prompt_sent))
    logging.info(f"Prompt sent to LLM saved to {prompt_file}")

    return result_file, prompt_file

# Function to save the costs
def save_llm_costs(costs: Dict[str, Any], query_dir: str, model: str):
    costs_file = os.path.join(query_dir, f"llm_costs_{model}.json")
    os.makedirs(os.path.dirname(costs_file), exist_ok=True)  # Ensure the directory exists
    with open(costs_file, 'w', encoding='utf-8') as f:
        json.dump(costs, f, ensure_ascii=False, indent=4)
    logging.info(f"LLM costs saved to {costs_file}")
    return costs_file

# Function to save the entire call_llm response
def save_call_llm_response(llm_response: Dict[str, Any], query_dir: str, model: str):
    response_file = os.path.join(query_dir, f"call_llm_response_{model}.txt")
    os.makedirs(os.path.dirname(response_file), exist_ok=True)  # Ensure the directory exists
    with open(response_file, 'w', encoding='utf-8') as f:
        f.write(str(llm_response))
    logging.info(f"call_llm response saved to {response_file}")
    return response_file

# Function to select the LLM function based on the model
def get_llm(provider: str, model_name: str):
    logging.info(f"Selecting LLM function for provider: {provider}, model: {model_name}")
    model_config = config['llm']['providers'][provider]['llm_models'][model_name]
    if provider == "openai":
        return OpenAILLM(config['llm']['assistant']['system_message'], model_config)
    elif provider == "vertex":
        return VertexLLM(config['llm']['assistant']['system_message'], model_config)
    elif provider == "anthropic":
        return AnthropicLLM(config['llm']['assistant']['system_message'], model_config)
    else:
        logging.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")

# Main function to process the LLM
def process_llm(query: str, top_chunks_csv: Optional[str], system_message: str, output_dir: str, llm_provider: str, llm_model: str, embedding_model: Optional[str], embeddings_csv: Optional[str] = None, top_k: int = 5):
    logging.info(f"Starting LLM processing for query: {query} with provider: {llm_provider}, model: {llm_model} and embeddings from {embedding_model}")
    
    query_hash = generate_query_hash(query)
    base_query_dir = os.path.join(output_dir, query_hash)
    os.makedirs(base_query_dir, exist_ok=True)
    logging.info(f"Created base query directory at {base_query_dir}")

    # Save the query for reference
    query_file = os.path.join(base_query_dir, "query.txt")
    with open(query_file, 'w', encoding='utf-8') as f:
        f.write(query)
    logging.info(f"Query saved to {query_file}")

    if not top_chunks_csv and embeddings_csv:
        logging.info(f"Generating top chunks for query: {query}")
        top_chunks_csv = os.path.join(base_query_dir, f"top_chunks_{embedding_model}.csv")
        process_query(
            query=query,
            embeddings_csv=embeddings_csv,
            top_k=top_k,
            output_dir=output_dir,
            model=embedding_model
        )
        logging.info(f"Top chunks generated and saved to {top_chunks_csv}")

    if not top_chunks_csv:
        logging.error("Either top_chunks_csv or embeddings_csv must be provided.")
        raise ValueError("Either top_chunks_csv or embeddings_csv must be provided.")

    # Create directory for the specific embedding and LLM model combination
    query_dir = os.path.join(base_query_dir, f"{embedding_model}_{llm_model}")
    os.makedirs(query_dir, exist_ok=True)
    logging.info(f"Created query directory for {embedding_model}_{llm_model} at {query_dir}")

    # Read top chunks for LLM processing
    logging.info(f"Reading top chunks from {top_chunks_csv} for LLM processing")
    top_chunks = pd.read_csv(top_chunks_csv).to_dict(orient='records')
    prompt_data = prepare_llm_prompt(query, top_chunks, system_message)
    save_prompt(prompt_data, query_dir, llm_model)
    
    llm = get_llm(llm_provider, llm_model)
    llm_response = llm.call_llm(prompt_data["text"])
    
    save_llm_result(llm_response["response_text"], prompt_data["text"], query_dir, llm_model)
    save_llm_costs(llm_response["costs"], query_dir, llm_model)
    save_call_llm_response(llm_response, query_dir, llm_model)

    logging.info(f"LLM processing completed for query: {query}")

    return llm_response
