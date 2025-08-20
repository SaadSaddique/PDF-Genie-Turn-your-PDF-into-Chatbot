# app/ingestion.py
import os, sys, uuid, traceback
from pypdf import PdfReader
from app.config import cfg
from app.chunkers.sentence import sentence_chunk
from app.chunkers.token import token_chunk
from app.embed.gemini_embed import GeminiEmbedder
from app.vector.chroma_store import ChromaStore

def parse_pdf(path: str):
    reader = PdfReader(path)
    pages = len(reader.pages)
    print(f"[ingestion] Opened {os.path.basename(path)} with {pages} pages")
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        yield i, text

def choose_chunker():
    if cfg.CHUNKER == "token":
        return lambda t: token_chunk(t, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    return lambda t: sentence_chunk(t, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)

def run_ingest(pdf_paths: list[str], collection: str = "pdf_rag", reset_collection: bool = False) -> int:
    """
    Ingest PDFs → chunk → embed → upsert to Chroma.
    Returns number of chunks written.
    If reset_collection=True, wipes the collection so ONLY current uploads remain.
    """
    os.makedirs(cfg.INDEX_DIR, exist_ok=True)
    print(f"[ingestion] INDEX_DIR={cfg.INDEX_DIR} • COLLECTION={collection} • RESET={reset_collection}")

    embedder = GeminiEmbedder()
    store = ChromaStore(collection, embedder)
    if reset_collection:
        print("[ingestion] Resetting collection …")
        store.reset_collection()

    splitter = choose_chunker()

    docs = []
    for pdf in pdf_paths:
        if not os.path.exists(pdf):
            raise FileNotFoundError(f"File not found: {pdf}")
        fname = os.path.basename(pdf)
        for page_num, page_text in parse_pdf(pdf):
            if not page_text:  # likely scanned page
                continue
            for chunk in splitter(page_text):
                c = (chunk or "").strip()
                if not c:
                    continue
                docs.append({
                    "id": str(uuid.uuid4()),
                    "text": c,
                    "meta": {"source": fname, "page": page_num}
                })

    if not docs:
        print("[ingestion] No text chunks found. Is the PDF scanned (image-only)?")
        return 0

    print(f"[ingestion] Adding {len(docs)} chunks to Chroma …")
    store.add(docs)
    print(f"[ingestion] ✅ Ingested {len(docs)} chunks. Index at: {cfg.INDEX_DIR}")
    return len(docs)

if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        if not args:
            print("Usage: python -m app.ingestion <pdf_path> [<pdf_path> ...]")
            sys.exit(1)
        n = run_ingest(args)
        print(f"Ingested {n} chunks.")
    except Exception as e:
        print("❌ Ingestion failed:", e)
        traceback.print_exc()
        sys.exit(2)
