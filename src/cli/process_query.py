# src/cli/process_query.py

import argparse
from app.process_query import process_query

def parse_args():
    parser = argparse.ArgumentParser(description='Process query to find similar document chunks')
    parser.add_argument('--query', type=str, required=True, help='The query string to search for')
    parser.add_argument('--embeddings_csv', type=str, required=True, help='Path to the CSV file with precomputed embeddings')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top similar chunks to retrieve')
    parser.add_argument('--result_dir', type=str, default='./data/result', help='Directory to save query results')
    parser.add_argument('--embedding_model', type=str, default='openai', choices=['openai', 'vertex'], help='Embedding model to use')
    return parser

def main(args):
    process_query(args.query, args.embeddings_csv, args.top_k, args.result_dir, args.embedding_model)

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    main(args)
