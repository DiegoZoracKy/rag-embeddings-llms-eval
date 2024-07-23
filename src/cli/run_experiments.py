import argparse
from experiments.run_experiments import run_experiments_from_config

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments based on YAML configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    return parser

def main(args):
    run_experiments_from_config(args.config)

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    main(args)
