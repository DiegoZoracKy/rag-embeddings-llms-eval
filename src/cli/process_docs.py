# src/cli/process_docs.py

import argparse
from app.process_docs import process_docs  # Import the core processing function

def parse_args():
    parser = argparse.ArgumentParser(description='Process documents to generate embeddings')
    group = parser.add_mutually_exclusive_group(required=True)  # Ensure only one of the arguments is required
    group.add_argument('--pdf_dir', type=str, help='Directory containing PDF files')
    group.add_argument('--pdf_file', type=str, help='File to be processed')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to store output files')
    parser.add_argument('--embedding_model', type=str, default='openai', choices=['openai', 'vertex'], help='Embedding model to use')
    return parser

def main(args):
    process_docs(args.output_dir, args.pdf_dir, args.pdf_file, args.embedding_model)

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    main(args)
