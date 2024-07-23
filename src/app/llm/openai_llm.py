# src/app/llm/openai_llm.py

import logging
from openai import OpenAI
from typing import Any, Dict
from .llm_base import LLMBase

class OpenAILLM(LLMBase):
    def __init__(self, system_message: str, model_config: Dict[str, Any]):
        self.system_message = system_message
        self.model_config = model_config
        self.prices = model_config['prices']
        self.client = OpenAI(api_key=self.model_config['api_key'])
        self.model = self.model_config['model']  # Store the current model in use
    
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        logging.info(f"Calling OpenAI LLM")
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.model_config.get('temperature', 0.7),
                max_tokens=self.model_config.get('max_tokens', 1000)
            )
            logging.info("Received response from OpenAI LLM")
            
            llm_response = {
                "prompt": prompt,
                "response_text": response.choices[0].message.content,
                "system_message": self.system_message,
                "response": response,
                "costs": self.calculate_cost(response)
            }
            return llm_response
        except Exception as e:
            logging.error(f"Error calling OpenAI LLM: {e}")
            raise
    
    def calculate_cost(self, response: Any) -> Dict[str, float]:
        total_tokens = response.usage.total_tokens
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        input_cost = input_tokens * self.prices['input_token']
        output_cost = output_tokens * self.prices['output_token']
        total_cost = input_cost + output_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }
