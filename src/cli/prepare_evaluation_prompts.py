import argparse
from app.llm.run_prepare_prompts import process_experiments

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare evaluation prompts for LLM evaluations')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing experiment data')
    return parser

def main(args):
    process_experiments(args.data_dir)

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    main(args)
