# src/cli/main.py

import sys
import argparse
import logging
from process_docs import main as process_docs_main, parse_args as process_docs_parse_args
from process_query import main as process_query_main, parse_args as process_query_parse_args
from process_llm import main as process_llm_main, parse_args as process_llm_parse_args
from prepare_evaluation_prompts import main as prepare_evaluation_prompts_main, parse_args as prepare_evaluation_prompts_parse_args
from prepare_groundedness_prompts import main as prepare_groundedness_prompts_main, parse_args as prepare_groundedness_prompts_parse_args
from run_groundedness_evaluators import main as run_groundedness_evaluators_main, parse_args as run_groundedness_evaluators_parse_args
from run_evaluators import main as run_evaluators_main, parse_args as run_evaluators_parse_args
from run_experiments import main as run_experiments_main, parse_args as run_experiments_parse_args

# Define available commands
COMMANDS = {
    'process-docs': (process_docs_main, process_docs_parse_args),
    'process-query': (process_query_main, process_query_parse_args),
    'process-llm': (process_llm_main, process_llm_parse_args),
    'prepare-evaluation-prompts': (prepare_evaluation_prompts_main, prepare_evaluation_prompts_parse_args),
    'prepare-groundedness-prompts': (prepare_groundedness_prompts_main, prepare_groundedness_prompts_parse_args),
    'run-groundedness-evaluators': (run_groundedness_evaluators_main, run_groundedness_evaluators_parse_args),
    'run-evaluators': (run_evaluators_main, run_evaluators_parse_args),
    'run-experiments': (run_experiments_main, run_experiments_parse_args),
}

def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(description='embeddings-llms-eval CLI')
    parser.add_argument('command', help='Command to run', choices=list(COMMANDS.keys()) + ['help'])
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help='Set the logging level')

    # Parse known arguments to separate top-level and subcommand arguments
    args, remaining_args = parser.parse_known_args()

    # Set logging level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True)

    # Display help for the overall CLI
    if args.command == 'help':
        parser.print_help()
    elif args.command in COMMANDS:
        command_main, command_parse_args = COMMANDS[args.command]

        # Now create a new parser for the subcommand and parse its arguments
        command_parser = command_parse_args()
        command_args = command_parser.parse_args(remaining_args)
        
        # Execute the command
        command_main(command_args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
