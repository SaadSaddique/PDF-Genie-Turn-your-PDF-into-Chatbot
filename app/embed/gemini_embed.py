import os
import google.generativeai as genai
from app.config import cfg
from .base import BaseEmbedder

class GeminiEmbedder(BaseEmbedder):
    """
    Uses Google's text-embedding-004.
    - embed_documents: for chunk embeddings (ingestion)
    - embed_query: for user query embedding (retrieval)
    """
    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or cfg.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
        genai.configure(api_key=key)
        self.model = model or cfg.GEMINI_EMBED_MODEL

    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            res = genai.embed_content(
                model=self.model,
                content=t,
                task_type="retrieval_document"
            )
            out.append(res["embedding"])
        return out

    def embed_query(self, text: str):
        res = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query"
        )
        return res["embedding"]

    # Back-compat: some callers may still use .embed()
    def embed(self, texts):
        return self.embed_documents(texts)
