# ui/streamlit_app.py
# --- Make the project root importable as "app" ---
import os, sys, shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from pypdf import PdfReader

from app.config import cfg
from app.ingestion import run_ingest
from app.chunkers.sentence import sentence_chunk
from app.chunkers.token import token_chunk
from app.query import ask
from app.embed.gemini_embed import GeminiEmbedder
from app.vector.chroma_store import ChromaStore

st.set_page_config(page_title="PDF Genie", layout="wide")
st.title("🧞 PDF Genie — Your Smart PDF Chatbot")

# ---------------- Helpers ----------------
def suggest_params_by_chunks(n_chunks: int):
    if n_chunks <= 50:
        return {"k": 6, "min_rel": 1.0, "label": "small corpus"}
    if n_chunks <= 300:
        return {"k": 8, "min_rel": 1.1, "label": "medium corpus"}
    return {"k": 10, "min_rel": 1.25, "label": "large corpus"}

def preview_chunks(pdf_paths, max_pages, chunker, size, overlap):
    total_pages, total_chunks = 0, 0
    chunk_fn = (lambda t: token_chunk(t, size, overlap)) if chunker == "token" else (lambda t: sentence_chunk(t, size, overlap))
    for pdf in pdf_paths:
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages, 1):
            if max_pages and i > max_pages:
                break
            text = (page.extract_text() or "").strip()
            total_pages += 1
            if not text:
                continue
            total_chunks += len(chunk_fn(text))
    return total_pages, total_chunks

def count_chunks_in_collection(collection: str) -> int:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=cfg.INDEX_DIR)
        col = client.get_or_create_collection(collection)
        return col.count()
    except Exception:
        return 0

# Session defaults for suggestions
if "top_k" not in st.session_state:
    st.session_state.top_k = cfg.TOP_K
if "min_rel" not in st.session_state:
    st.session_state.min_rel = 1.0

# ---------------- Sidebar ----------------
st.sidebar.header("Retrieval Settings")
top_k = st.sidebar.slider("Top‑K", 1, 12, st.session_state.top_k)
min_rel = st.sidebar.number_input(
    "Min relevance (max distance; lower = stricter)",
    min_value=0.0, max_value=1.5, value=float(st.session_state.min_rel), step=0.05,
    help="Chroma uses cosine distance: lower is better. Results with distance > this are dropped.",
)
chunker_choice = st.sidebar.selectbox("Chunker", ["sentence", "token"], index=0 if cfg.CHUNKER == "sentence" else 1)
chunk_size = st.sidebar.number_input("Chunk size", 100, 2000, cfg.CHUNK_SIZE, 50)
chunk_overlap = st.sidebar.number_input("Chunk overlap", 0, 500, cfg.CHUNK_OVERLAP, 10)
collection = st.sidebar.text_input("Collection name", "pdf_rag")
st.sidebar.caption("Tip: preview before indexing to control cost.")

# ---------------- Upload & Index ----------------
st.header("1) Upload & Index PDFs")
files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
max_pages = st.number_input("Cap pages per file (0 = all)", min_value=0, max_value=2000, value=0, step=1)
simulate = st.checkbox("Preview chunks (no API calls)", value=True)

paths = []
if files:
    os.makedirs("data/raw_pdfs", exist_ok=True)

    # ALWAYS clear raw_pdfs so only current files exist
    for name in os.listdir("data/raw_pdfs"):
        p = os.path.join("data/raw_pdfs", name)
        try:
            os.remove(p)
        except IsADirectoryError:
            shutil.rmtree(p, ignore_errors=True)

    # Save current uploads
    for f in files:
        p = os.path.join("data/raw_pdfs", f.name)
        with open(p, "wb") as out:
            out.write(f.read())
        paths.append(p)

    if simulate:
        pages, chunks = preview_chunks(paths, max_pages or None, chunker_choice, int(chunk_size), int(chunk_overlap))
        st.info(f"Preview: ~{pages} pages → ~{chunks} chunks (no API calls yet).")
        s = suggest_params_by_chunks(chunks)
        st.caption(f"Suggested (preview: {s['label']}): **Top‑K ≈ {s['k']}**, **Min relevance ≈ {s['min_rel']}**.")
        if st.button("Apply suggested retrieval params (from preview)"):
            st.session_state.top_k = s["k"]
            st.session_state.min_rel = s["min_rel"]
            st.experimental_rerun()

    if st.button("Index now"):
        cfg.CHUNKER = chunker_choice
        cfg.CHUNK_SIZE = int(chunk_size)
        cfg.CHUNK_OVERLAP = int(chunk_overlap)
        with st.spinner(f"Embedding chunks and writing to the '{collection}' index…"):
            # ALWAYS reset collection so index only contains these uploads
            n = run_ingest(paths, collection=collection, reset_collection=True)
        if n == 0:
            st.error("Indexed 0 chunks (likely image-only/scanned PDF). Try a text PDF or add OCR.")
        else:
            st.success(f"✅ Indexed {n} chunks into collection '{collection}'.")
            total = count_chunks_in_collection(collection)
            if total > 0:
                s = suggest_params_by_chunks(total)
                st.caption(
                    f"Index size: **{total} chunks** ({s['label']}). Suggested: **Top‑K ≈ {s['k']}**, **Min relevance ≈ {s['min_rel']}**."
                )
                if st.button("Apply suggested retrieval params (from index)"):
                    st.session_state.top_k = s["k"]
                    st.session_state.min_rel = s["min_rel"]
                    st.experimental_rerun()

# ---------------- Ask ----------------
st.header("2) Ask a Question")
q = st.text_input("Your question", placeholder="e.g., Summarize this document with [n] citations.")
ask_clicked = st.button("Ask")

if ask_clicked:
    if not q.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Retrieving context & generating answer…"):
            ans = ask(q, top_k=int(top_k), collection=collection, min_relevance=float(min_rel))

        st.subheader("Answer")
        st.markdown(ans.answer)  # natural chat text only

        # Hidden-by-default sources
        if st.button("Show sources & snippets"):
            try:
                store = ChromaStore(collection, GeminiEmbedder())
                ids = [c.chunk_id for c in ans.citations]
                id2txt = store.get_texts_by_ids(ids)
                st.write("**Sources**")
                for i, c in enumerate(ans.citations, 1):
                    snippet = (id2txt.get(c.chunk_id, "") or "").strip().replace("\n", " ")
                    if len(snippet) > 300:
                        snippet = snippet[:300] + "..."
                    st.write(f"[{i}] {c.source} (p.{c.page}) • score={c.score}")
                    st.caption(snippet if snippet else "(snippet unavailable)")
            except Exception as e:
                st.error(f"Could not fetch snippets: {e}")

        # Retrieval debug (optional)
        with st.expander("Retrieval Debug"):
            try:
                store = ChromaStore(collection, GeminiEmbedder())
                hits = store.query(q, k=int(top_k), min_relevance=float(min_rel))
                if not hits:
                    st.warning("No chunks retrieved. Try increasing Top‑K or raising Min relevance (e.g., 1.2).")
                else:
                    for i, h in enumerate(hits, 1):
                        meta = h.get("meta") or {}
                        dist = h.get("score")
                        st.write(
                            f"[{i}] id={h['id']} • page={meta.get('page')} • distance={dist:.3f} • source={meta.get('source')}"
                        )
                        snippet = (h["text"] or "").strip().replace("\n", " ")
                        if len(snippet) > 240:
                            snippet = snippet[:240] + "..."
                        st.caption(snippet)
            except Exception as e:
                st.error(f"Debug retrieval failed: {e}")
