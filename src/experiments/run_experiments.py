# src/experiments/run_experiments.py

import sys
import os
import yaml
import argparse
import itertools
import logging
from typing import List, Dict
from app.process_llm import process_llm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True)

BASE_DIR = "./data/experiments"

# Load the configuration file
def load_config(config_path: str) -> Dict:
    logging.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info("Configuration loaded successfully")
            return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise

# Generate combinations of parameter variations
def generate_combinations(params_variations: Dict[str, List]) -> List[Dict[str, str]]:
    logging.info("Generating combinations of parameter variations")
    keys = params_variations.keys()
    values = [params_variations[key] for key in keys]
    for combination in itertools.product(*values):
        logging.debug(f"Generated combination: {combination}")
        yield dict(zip(keys, combination))

# Run an individual experiment
def run_experiment(query: str, system_message: str, top_k: int, params: Dict, experiment_id: str):
    logging.info(f"Running experiment {experiment_id}")
    
    try:
        experiment_dir = os.path.join(BASE_DIR, experiment_id)
        llm_provider = params['llm_model']['provider']
        llm_model = params['llm_model']['model']
        embedding_model = params['embeddings']['model']
        run_dir = os.path.join(experiment_dir, f"{experiment_id}_{embedding_model}_{llm_model}")
        os.makedirs(run_dir, exist_ok=True)

        # Save query.txt at the root of the experiment directory
        query_file = os.path.join(experiment_dir, "query.txt")
        with open(query_file, 'w') as f:
            f.write(query)
        logging.info(f"Query saved to {query_file}")

        params_file = os.path.join(run_dir, "params.yaml")
        with open(params_file, 'w') as f:
            yaml.dump(params, f)
        logging.info(f"Parameters saved to {params_file}")

        embeddings_csv = params['embeddings']['csv']

        response = process_llm(
            query=query,
            top_chunks_csv=None,  # No top_chunks_csv provided, should be generated
            system_message=system_message,
            output_dir=run_dir,
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_model=embedding_model,
            embeddings_csv=embeddings_csv,
            top_k=top_k
        )
        
        logging.info(f"Experiment {experiment_id} completed successfully")
    except Exception as e:
        logging.error(f"Error in experiment {experiment_id}: {e}")

# Run all experiments based on the configuration file
def run_experiments_from_config(config_path: str):
    logging.info(f"Starting to run experiments from {config_path}")
    try:
        config = load_config(config_path)
        experiments = config['experiments']

        for experiment_id, experiment_config in experiments.items():
            logging.info(f"Setting up experiment {experiment_id}")
            params_fixed = experiment_config['params_fixed']
            params_variations = experiment_config['params_variations']

            query = params_fixed['query']
            system_message = params_fixed['system_message']
            top_k = params_fixed['top_k']

            # Ensure the query.txt is saved at the root of the experiment directory
            experiment_dir = os.path.join(BASE_DIR, experiment_id)
            os.makedirs(experiment_dir, exist_ok=True)
            query_file = os.path.join(experiment_dir, "query.txt")
            with open(query_file, 'w') as f:
                f.write(query)
            logging.info(f"Query saved to {query_file}")

            for params in generate_combinations(params_variations):
                run_experiment(query, system_message, top_k, params, experiment_id)
        logging.info("All experiments completed successfully")
    except Exception as e:
        logging.error(f"Failed to run experiments: {e}")
        raise
