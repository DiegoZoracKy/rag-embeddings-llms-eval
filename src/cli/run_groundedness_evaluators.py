import argparse
from app.llm.run_groundedness_evaluators import run_groundedness_evaluators

def parse_args():
    parser = argparse.ArgumentParser(description='Run groundedness evaluators for LLM responses')
    parser.add_argument('--evaluation_dir', type=str, required=True, help='Directory containing evaluation data')
    return parser

def main(args):
    run_groundedness_evaluators(args.evaluation_dir)

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    main(args)
