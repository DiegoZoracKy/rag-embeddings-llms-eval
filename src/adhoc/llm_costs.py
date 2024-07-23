import os
import sys
import logging
from typing import Any, List, Dict
from app.llm.openai_llm import OpenAILLM
from app.llm.vertex_llm import VertexLLM
from app.llm.anthropic_llm import AnthropicLLM
from app.llm.run_prepare_prompts import read_file
from utils.config_utils import load_config

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True)

# Load configuration
config = load_config()
if config is None:
    raise SystemExit("Exiting due to missing configuration.")

def main():
    # # Instantiate LLM Client
    vertex_llm = VertexLLM(config['llm']['evaluator'].get('system_message'), config['llm']['models']['vertex'])
    openai_llm = VertexLLM(config['llm']['evaluator'].get('system_message'), config['llm']['models']['openai'])

    # # Prompt
    # prompt = "1+1=?"

    # # Evaluate using Vertex LLM
    # vertex_response = vertex_llm.call_llm(prompt)
    # print(f"==== Vertex ====")
    # print(vertex_response)
    # vertex_cost = vertex_llm.calculate_cost(vertex_response, len(prompt))
    # print("==== Cost ====")
    # print(vertex_cost)

    # Evaluate using OpenAI LLM
    openai_response = openai_llm.call_llm(prompt)
    print(f"==== OpenAI ====")
    print(openai_response)
    openai_cost = openai_llm.calculate_cost(openai_response)
    print("==== Cost ====")
    print(openai_cost)

    # # Evaluate using Anthropic LLM
    # anthropic_response = anthropic_llm.call_llm(prompt)
    # print(f"==== Anthropic ====")
    # print(anthropic_response)
    # anthropic_cost = anthropic_llm.calculate_cost(anthropic_response)
    # print("==== Cost ====")
    # print(anthropic_cost)
   
if __name__ == "__main__":
    main()