# src/app/llm/vertex_llm.py

import logging
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import (
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
)
from typing import Any, Dict
from .llm_base import LLMBase

class VertexLLM(LLMBase):
    def __init__(self, system_message: str, model_config: Dict[str, Any]):
        self.system_message = system_message
        self.model_config = model_config
        self.prices = model_config['prices']
        aiplatform.init(project=model_config['project_id'], location=model_config['location'])
        self.client = GenerativeModel(model_config['model'], system_instruction=[self.system_message])
        self.model = self.model_config['model']  # Store the current model in use
    
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        logging.info(f"Calling Vertex LLM")
        try:
            response = self.client.generate_content(
                contents=[prompt],
                generation_config=GenerationConfig(
                    temperature=self.model_config.get('temperature', 0.7),
                    max_output_tokens=self.model_config.get('max_output_tokens', 1000)
                ),
                stream=False,
                safety_settings=self.model_config.get('safety_settings', {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                })
            )
            logging.info("Received response from Vertex LLM")
            
            llm_response = {
                "prompt": prompt,
                "response_text": response.candidates[0].content.parts[0].text,
                "system_message": self.client._system_instruction,
                "response": response,
                "costs": self.calculate_cost(response, len(prompt))
            }
            return llm_response
        except Exception as e:
            logging.error(f"Error calling Vertex LLM: {e}")
            raise
    
    def calculate_cost(self, response: Any, prompt_length: int) -> Dict[str, float]:
        total_tokens = response.usage_metadata.total_token_count
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count

        if prompt_length > 128000:
            input_cost = input_tokens * self.prices['input_token_long']
            output_cost = output_tokens * self.prices['output_token_long']
        else:
            input_cost = input_tokens * self.prices['input_token_short']
            output_cost = output_tokens * self.prices['output_token_short']

        total_cost = input_cost + output_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }
