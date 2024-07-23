# src/cli/process_llm.py

import argparse
from app.process_llm import process_llm
from utils.config_utils import load_config
import os

# Load configuration
config = load_config()

def parse_args():
    parser = argparse.ArgumentParser(description='Process LLM for a query')
    parser.add_argument('--query', type=str, required=True, help='The query string to search for')
    parser.add_argument('--top_chunks_csv', type=str, required=False, help='Path to the CSV file with top chunks')
    parser.add_argument('--system_message', type=str, default=config.get('llm', {}).get('system_message', 'Default system message'), help='System message for the LLM')
    parser.add_argument('--output_dir', type=str, default='./data/result', help='Directory to save LLM results')
    parser.add_argument('--llm_model', type=str, default=config.get('llm', {}).get('default', 'openai'), choices=['openai', 'vertex', 'anthropic'], help='LLM model to use')
    parser.add_argument('--embedding_model', type=str, default=config.get('embeddings', {}).get('default', 'openai'), choices=['openai', 'vertex'], help='Embedding model to use')
    parser.add_argument('--embeddings_csv', type=str, required=False, help='Path to the embeddings CSV file for similarity search')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top chunks to retrieve')
    return parser

def main(args):
    process_llm(
        query=args.query,
        top_chunks_csv=args.top_chunks_csv,
        system_message=args.system_message,
        output_dir=args.output_dir,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        embeddings_csv=args.embeddings_csv,
        top_k=args.top_k
    )

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    main(args)
