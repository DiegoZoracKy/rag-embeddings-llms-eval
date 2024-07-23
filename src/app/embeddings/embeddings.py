# src/app/embeddings/embeddings.py

from abc import ABC, abstractmethod
from typing import List

class EmbeddingGenerator(ABC):
    @abstractmethod
    def generate(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            List[float]: The generated embedding.
        """
        pass
