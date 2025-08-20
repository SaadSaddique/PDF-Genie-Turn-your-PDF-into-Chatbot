from nltk.tokenize import sent_tokenize

def sentence_chunk(text: str, chunk_size_chars=800, overlap=120):
    sents = sent_tokenize(text or "")
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= chunk_size_chars:
            cur += (" " if cur else "") + s
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)

    if overlap <= 0 or not chunks:
        return chunks

    with_overlap = []
    for i, c in enumerate(chunks):
        prev_tail = chunks[i-1][-overlap:] if i > 0 else ""
        with_overlap.append(prev_tail + c)
    return with_overlap
