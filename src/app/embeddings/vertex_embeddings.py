import logging
import numpy as np
from typing import List
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from app.embeddings.embeddings import EmbeddingGenerator
from utils.config_utils import load_config

class VertexEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, project_id: str, location: str, model_name: str = "text-embedding-004"):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.location)
        self.text_embedding_model = TextEmbeddingModel.from_pretrained(self.model_name)

    def generate(self, text: str, return_array: bool = False) -> List[float]:
        try:
            embeddings = self.text_embedding_model.get_embeddings([text])
            text_embedding = [embedding.values for embedding in embeddings][0]

            if return_array:
                text_embedding = np.fromiter(text_embedding, dtype=float)

            # Returns 768-dimensional array or list
            return text_embedding
        except Exception as e:
            logging.error(f"Error generating embedding with Vertex AI: {e}")
            return []
