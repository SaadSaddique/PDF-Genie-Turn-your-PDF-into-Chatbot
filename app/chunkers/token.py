import tiktoken

# Uses OpenAI's cl100k_base tokenizer as a practical default
def token_chunk(text: str, chunk_size=800, overlap=120):
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text or "")
    out = []
    start = 0
    while start < len(toks):
        end = min(start + chunk_size, len(toks))
        segment = enc.decode(toks[start:end])
        out.append(segment)
        start = end - overlap
        if start < 0:
            start = 0
    return out
