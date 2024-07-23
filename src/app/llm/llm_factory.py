from typing import Any, Dict
from llm_base import LLMBase
from utils.config_utils import load_config
from openai_llm import OpenAILLM
from vertex_llm import VertexLLM

class LLMFactory:
    @staticmethod
    def create_llm(model_type: str, system_message: str, model_config: Dict[str, Any]) -> LLMBase:
        if model_type == 'openai':
            return OpenAILLM(system_message, model_config)
        elif model_type == 'vertex':
            return VertexLLM(system_message, model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

class Evaluator:
    def __init__(self, model_type: str, model_config: Dict[str, Any]):
        config = load_config()
        system_message = config['llm']['evaluator'].get('system_message')
        self.llm = LLMFactory.create_llm(model_type, system_message, model_config)

    def evaluate(self, prompt: str) -> str:
        return self.llm.call_llm(prompt)

class Assistant:
    def __init__(self, model_type: str, model_config: Dict[str, Any]):
        config = load_config()
        system_message = config['llm']['assistant'].get('system_message')
        self.llm = LLMFactory.create_llm(model_type, system_message, model_config)

    def assist(self, prompt: str) -> str:
        return self.llm.call_llm(prompt)
