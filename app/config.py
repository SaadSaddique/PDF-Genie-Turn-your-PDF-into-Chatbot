from dotenv import load_dotenv
import os

load_dotenv()

class Cfg:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "gemini")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")
    GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

    VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")
    INDEX_DIR = os.getenv("INDEX_DIR", "./data/index")

    CHUNKER = os.getenv("CHUNKER", "sentence")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

    TOP_K = int(os.getenv("TOP_K", "5"))
    USE_MMR = os.getenv("USE_MMR", "true").lower() == "true"

    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

cfg = Cfg()
