# src/app/llm/run_groundedness_evaluators.py

import os
import sys
import logging
from typing import List, Dict
from app.llm.openai_llm import OpenAILLM
from app.llm.vertex_llm import VertexLLM
from app.llm.anthropic_llm import AnthropicLLM
from app.llm.run_prepare_prompts import read_file
from utils.config_utils import load_config

def run_groundedness_evaluators(evaluation_dir: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True)

    # Load configuration
    config = load_config()
    if config is None:
        raise SystemExit("Exiting due to missing configuration.")

    # Define the providers and models you want to evaluate
    evaluators = {
        "vertex": ["gemini-1.5-pro-001"],
        "openai": ["gpt-4o-2024-05-13"],
        "anthropic": ["claude-3-5-sonnet-20240620"]
    }

    def instantiate_evaluator(provider: str, model_name: str):
        model_config = config['llm']['providers'][provider]['llm_models'][model_name]
        if provider == "openai":
            return OpenAILLM(config['llm']['groundedness_evaluator'].get('system_message'), model_config)
        elif provider == "vertex":
            return VertexLLM(config['llm']['groundedness_evaluator'].get('system_message'), model_config)
        elif provider == "anthropic":
            return AnthropicLLM(config['llm']['groundedness_evaluator'].get('system_message'), model_config)
        else:
            logging.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def read_all_groundedness_prompts(evaluation_dir: str) -> List[Dict[str, str]]:
        all_prompts = []
        for root, dirs, files in os.walk(evaluation_dir):
            for file in files:
                if file.startswith("groundedness_prompt_") and file.endswith(".txt"):
                    model = file.split("groundedness_prompt_")[1].split(".txt")[0]
                    prompt = read_file(os.path.join(root, file))
                    experiment_dir = os.path.relpath(root, evaluation_dir)
                    all_prompts.append({
                        "experiment_dir": experiment_dir,
                        "model": model,
                        "prompt": prompt
                    })
        return all_prompts

    all_prompts = read_all_groundedness_prompts(evaluation_dir)

    for evaluation in all_prompts:
        experiment_dir = evaluation["experiment_dir"]
        model = evaluation["model"]
        prompt = evaluation["prompt"]

        print("="*80)
        print(f"Prompt from {experiment_dir} for model {model}:\n")

        for provider, models in evaluators.items():
            for model_name in models:
                evaluator = instantiate_evaluator(provider, model_name)
                response = evaluator.call_llm(prompt)
                response_file = os.path.join(evaluation_dir, experiment_dir, f"groundedness_evaluation_{provider}_{model_name}.txt")
                os.makedirs(os.path.dirname(response_file), exist_ok=True)
                with open(response_file, 'w') as f:
                    f.write(response["response_text"])
                print(f"==== Groundedness Evaluation ({provider}_{model_name}) ====")
                print(response["response_text"])
