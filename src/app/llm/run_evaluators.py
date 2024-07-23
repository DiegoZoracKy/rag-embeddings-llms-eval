import os
import sys
import logging
from typing import List
from .openai_llm import OpenAILLM
from .run_prepare_prompts import read_all_evaluation_prompts
from .vertex_llm import VertexLLM
from .anthropic_llm import AnthropicLLM
from utils.config_utils import load_config

def run_evaluators(evaluation_dir: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True)

    # Load configuration
    config = load_config()
    if config is None:
        raise SystemExit("Exiting due to missing configuration.")

    # Function to instantiate evaluators based on the provider and model
    def get_evaluator(provider: str, model_name: str):
        logging.info(f"Selecting evaluator for provider: {provider}, model: {model_name}")
        model_config = config['llm']['providers'][provider]['llm_models'][model_name]
        if provider == "openai":
            return OpenAILLM(config['llm']['evaluator']['system_message'], model_config)
        elif provider == "vertex":
            return VertexLLM(config['llm']['evaluator']['system_message'], model_config)
        elif provider == "anthropic":
            return AnthropicLLM(config['llm']['evaluator']['system_message'], model_config)
        else:
            logging.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")

    # Define the providers and models you want to evaluate
    evaluators = {
        "vertex": ["gemini-1.5-pro-001"],
        "openai": ["gpt-4o-2024-05-13"],
        "anthropic": ["claude-3-5-sonnet-20240620"]
    }

    all_prompts = read_all_evaluation_prompts(evaluation_dir)

    for evaluation in all_prompts:
        experiment_dir = evaluation["experiment_dir"]
        prompt = evaluation["prompt"]
        
        print(f"Prompt from {experiment_dir}:\n")

        for provider, models in evaluators.items():
            for model_name in models:
                evaluator = get_evaluator(provider, model_name)
                response = evaluator.call_llm(prompt)
                response_file = os.path.join(evaluation_dir, experiment_dir, f"llm_response_{provider}_{model_name}.txt")
                with open(response_file, 'w') as f:
                    f.write(response["response_text"])
                print(f"==== {provider.capitalize()} ({model_name}) Evaluation ====")
                print(response["response_text"])
