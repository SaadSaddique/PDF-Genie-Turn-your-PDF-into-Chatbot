# app/prompt.py

def build_prompt(ctx_blocks, user_q: str) -> str:
    """
    Build a citation-aware prompt that asks for a natural Markdown answer.
    We enumerate context blocks as [1]..[k]. The model should cite [n] when
    statements are supported by a block. If context is insufficient, it must say so.
    """
    numbered = []
    for i, b in enumerate(ctx_blocks, 1):
        header = f"[{i}] (source: {b['source']}, page: {b['page']}, id: {b['id']}, score: {b['score']})"
        numbered.append(header + "\n" + (b["text"] or ""))

    context = "\n\n".join(numbered)

    return f"""
You are a careful assistant performing Retrieval‑Augmented Generation (RAG).

RULES
- Use ONLY the CONTEXT blocks below. Do not invent facts.
- If the context is insufficient to answer, say so clearly.
- Prefer comprehensive coverage over focusing on a single snippet.
- Cite using bracketed indices like [1], [2], referring to the CONTEXT blocks used.
- Answer in clear, concise **Markdown** (no JSON in the final answer).

CONTEXT
{context}

USER QUESTION
{user_q}

REPLY
Write the answer in Markdown with [n] citations where appropriate.
"""
