import logging
from typing import List
from openai import OpenAI
from app.embeddings.embeddings import EmbeddingGenerator

class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
                encoding_format="float"  # Use "float" or "base64"
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return []
