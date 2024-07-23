# src/app/llm/llm_base.py

from abc import ABC, abstractmethod

class LLMBase(ABC):
    @abstractmethod
    def call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the given prompt.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass
