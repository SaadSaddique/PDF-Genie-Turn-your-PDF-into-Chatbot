import google.generativeai as genai
from app.config import cfg
from .base import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        genai.configure(api_key=api_key or cfg.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model or cfg.GEMINI_MODEL)

    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.2) -> str:
        resp = self.model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature
            }
        )
        return resp.text or ""
