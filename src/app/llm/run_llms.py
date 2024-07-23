import os
from typing import List
from app.llm.openai_llm import OpenAILLM
from app.llm.vertex_llm import VertexLLM
from app.llm.llm_factory import LLMFactory
from utils.config_utils import load_config

# Load configuration
config = load_config()
if config is None:
    raise SystemExit("Exiting due to missing configuration.")

# Define Evaluator class as per previous definitions
class Evaluator:
    def __init__(self, model_type: str, model_config: dict):
        system_message = config['llm'].get('system_message', "Você é um avaliador de experimentos. Compare os resultados dos experimentos e forneça uma avaliação detalhada.")
        self.llm = LLMFactory.create_llm(model_type, system_message, model_config)

    def evaluate(self, prompt: str) -> str:
        return self.llm.call_llm(prompt)

# Instantiate Evaluators
openai_evaluator = Evaluator('openai', config['llm']['models']['openai'])
vertex_evaluator = Evaluator('vertex', config['llm']['models']['vertex'])

# List of prompts to be evaluated
prompts = [
    "Avalie os resultados dos experimentos de IA.",
    "Compare os desempenhos dos modelos OpenAI e Vertex.",
    "Quais são as principais diferenças entre os resultados obtidos?"
]

# Function to save text to a file
def save_text(file_path: str, text: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(text)

# Main processing loop
for index, prompt in enumerate(prompts):
    # Call both LLMs
    openai_response = openai_evaluator.evaluate(prompt)
    vertex_response = vertex_evaluator.evaluate(prompt)

    # Define file paths
    base_dir = os.path.join("data", "evaluations", str(index))
    openai_response_path = os.path.join(base_dir, "llm_response_openai.txt")
    vertex_response_path = os.path.join(base_dir, "llm_response_vertex.txt")
    openai_prompt_path = os.path.join(base_dir, "prompt_openai.txt")
    vertex_prompt_path = os.path.join(base_dir, "prompt_vertex.txt")

    # Save the responses and prompts
    save_text(openai_response_path, openai_response)
    save_text(vertex_response_path, vertex_response)
    save_text(openai_prompt_path, prompt)
    save_text(vertex_prompt_path, prompt)

print("Processing completed successfully.")
