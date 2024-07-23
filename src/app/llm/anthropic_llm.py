# src/app/llm/anthropic_llm.py

import logging
from typing import Any, Dict
import anthropic
from .llm_base import LLMBase

class AnthropicLLM(LLMBase):
    def __init__(self, system_message: str, model_config: Dict[str, Any]):
        self.system_message = system_message
        self.model_config = model_config
        self.prices = model_config['prices']
        self.client = anthropic.Anthropic(api_key=self.model_config['api_key'])
        self.model = self.model_config['model']  # Store the current model in use
    
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        logging.info(f"Calling Anthropic LLM")
        messages = [
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.messages.create(
                model=self.model,
                system=self.system_message,
                messages=messages,
                max_tokens=self.model_config.get('max_tokens', 1000),
                temperature=self.model_config.get('temperature', 0.7)
            )
            logging.info("Received response from Anthropic LLM")
            
            # Concatenate the text blocks into a single string
            response_text = ''.join(block.text for block in response.content)

            llm_response = {
                "prompt": prompt,
                "response_text": response_text,
                "system_message": self.system_message,
                "response": response,
                "costs": self.calculate_cost(response)
            }
            return llm_response
        except Exception as e:
            logging.error(f"Error calling Anthropic LLM: {e}")
            raise
    
    def calculate_cost(self, response: Any) -> Dict[str, float]:
        total_tokens = response.usage.input_tokens + response.usage.output_tokens
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

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
