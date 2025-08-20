# app/query.py
import re
from typing import Optional, List, Dict, Any
from app.config import cfg
from app.embed.gemini_embed import GeminiEmbedder
from app.vector.chroma_store import ChromaStore
from app.prompt import build_prompt
from app.llm.gemini import GeminiLLM
from app.schema import Answer, Citation  # keep using your Pydantic container

SUMMARY_PATTERNS = [
    r"\bsummar(y|ise|ize)\b", r"\boverview\b", r"\bwhat('s| is) this (pdf|document) (about)\b",
    r"\btl;dr\b", r"\bhigh\-level\b", r"\babstract\b"
]

def is_summary_like(q: str) -> bool:
    ql = (q or "").lower()
    return any(re.search(p, ql) for p in SUMMARY_PATTERNS)

def markers_used(answer_text: str) -> List[int]:
    # returns unique [n] markers in order of first appearance
    nums = re.findall(r"\[(\d+)\]", answer_text or "")
    seen, out = set(), []
    for n in nums:
        i = int(n)
        if i not in seen:
            seen.add(i); out.append(i)
    return out

def retrieve(query: str, k: int, *, collection: str = "pdf_rag",
             min_relevance: Optional[float] = None) -> List[Dict[str, Any]]:
    store = ChromaStore(collection, GeminiEmbedder())
    results = store.query(query, k=k, min_relevance=min_relevance)  # page-diversified in store
    blocks = []
    for r in results:
        m = r["meta"] or {}
        blocks.append({
            "id": r["id"],
            "text": r["text"],
            "source": m.get("source", "unknown"),
            "page": m.get("page", None),
            "score": r.get("score"),
        })
    return blocks

def ask(user_q: str, *, top_k: Optional[int] = None, collection: str = "pdf_rag",
        min_relevance: Optional[float] = None) -> Answer:
    """
    Returns a natural-language Answer.answer (Markdown) with hidden-by-default citations list.
    No JSON is requested from the model anymore; we parse [n] markers to map citations.
    """
    # Smart retrieval knobs
    summary_mode = is_summary_like(user_q)
    k = top_k if top_k is not None else cfg.TOP_K
    if summary_mode:
        k = max(k, 10)                  # pull more context for summaries
        min_relevance = max(min_relevance or 1.0, 1.2)  # loosen distance filter for breadth

    ctx = retrieve(user_q, k, collection=collection, min_relevance=min_relevance)

    if not ctx:
        return Answer(
            answer=("I couldn’t retrieve any relevant context from your indexed documents. "
                    "Please confirm you clicked **Index now**, the **collection** is correct, "
                    "and try increasing **Top‑K** or raising **Min relevance (max distance)**."),
            citations=[],
            confidence=0.0
        )

    # Build prompt → LLM
    prompt = build_prompt(ctx, user_q)
    llm = GeminiLLM()
    raw = llm.generate(prompt, cfg.MAX_TOKENS, cfg.TEMPERATURE)

    # Align citations to markers in the answer
    used = markers_used(raw)
    if used:
        selected = []
        for idx in used:
            if 1 <= idx <= len(ctx):
                b = ctx[idx - 1]
                selected.append(Citation(
                    source=b["source"], page=b["page"], chunk_id=b["id"], score=b["score"]
                ))
        citations = selected
    else:
        citations = []  # keep hidden unless user asks; we can fall back to top‑k if you prefer

    return Answer(answer=raw, citations=citations, confidence=None)

if __name__ == "__main__":
    import sys, json
    q = " ".join(sys.argv[1:]) or "What is this document about?"
    print(ask(q).model_dump_json(indent=2))
