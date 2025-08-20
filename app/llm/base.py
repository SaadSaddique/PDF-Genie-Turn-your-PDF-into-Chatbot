from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.2) -> str:
        ...
