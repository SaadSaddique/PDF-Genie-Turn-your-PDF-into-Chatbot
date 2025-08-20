# app/vector/chroma_store.py
import chromadb
from collections import defaultdict
from app.config import cfg

class ChromaStore:
    def __init__(self, collection_name: str, embedder):
        self.client = chromadb.PersistentClient(path=cfg.INDEX_DIR)
        self.collection_name = collection_name
        self.col = self.client.get_or_create_collection(name=self.collection_name)
        self.embedder = embedder

    # ---- lifecycle ----
    def reset_collection(self):
        """Drop and recreate the current collection (fresh, empty)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # ok if it didn't exist
        self.col = self.client.get_or_create_collection(name=self.collection_name)

    def count(self) -> int:
        try:
            return self.col.count()
        except Exception:
            return 0

    # ---- writes ----
    def add(self, docs: list[dict]):
        """
        docs: [{"id": str, "text": str, "meta": dict}]
        We embed documents explicitly and pass vectors to Chroma.
        """
        texts = [d["text"] for d in docs]
        embeddings = self.embedder.embed_documents(texts)
        self.col.add(
            ids=[d["id"] for d in docs],
            documents=texts,
            metadatas=[d["meta"] for d in docs],
            embeddings=embeddings,
        )

    # ---- reads ----
    def _diversify_by_page(self, items: list[dict], k: int, per_page_cap: int = 2) -> list[dict]:
        """Round‑robin across pages so we cover the document broadly."""
        by_page: dict[int | None, list[dict]] = defaultdict(list)
        for it in items:
            page = (it.get("meta") or {}).get("page", None)
            by_page[page].append(it)

        selected: list[dict] = []
        while len(selected) < k:
            progressed = False
            for page in sorted(by_page.keys(), key=lambda x: (x is None, x)):
                bucket = by_page[page]
                if not bucket:
                    continue
                taken_here = sum(1 for s in selected if (s.get("meta") or {}).get("page") == page)
                if taken_here >= per_page_cap:
                    continue
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) >= k:
                    break
            if not progressed:
                break
        return selected[:k]

    def query(
        self,
        q: str,
        k: int = 5,
        min_relevance: float | None = None,
        diversify: bool = True,
        per_page_cap: int = 2,
    ) -> list[dict]:
        """
        Returns list of dicts: {id, text, meta, score}
        - Chroma returns cosine *distance* (lower is better; 0 = identical).
        - If min_relevance is set, drop results whose distance > min_relevance.
        - Over‑fetch (k*4) to allow filtering + diversification, then sort by distance.
        """
        # embed query
        if hasattr(self.embedder, "embed_query"):
            q_emb = self.embedder.embed_query(q)
        else:
            q_emb = self.embedder.embed([q])[0]

        n_results = max(k * 4, k)
        res = self.col.query(query_embeddings=[q_emb], n_results=n_results)

        items: list[dict] = []
        for doc, meta, _id, dist in zip(
            res.get("documents", [[]])[0],
            res.get("metadatas", [[]])[0],
            res.get("ids", [[]])[0],
            res.get("distances", [[]])[0],
        ):
            d = float(dist)
            if (min_relevance is not None) and (d > float(min_relevance)):
                continue
            items.append({"id": _id, "text": doc, "meta": meta, "score": d})

        items.sort(key=lambda x: x["score"])

        if diversify and items:
            return self._diversify_by_page(items, k=k, per_page_cap=per_page_cap)
        return items[:k]

    def get_texts_by_ids(self, ids: list[str]) -> dict[str, str]:
        """Fetch documents by id → {id: text}"""
        if not ids:
            return {}
        got = self.col.get(ids=list(ids))
        return { _id: doc for _id, doc in zip(got.get("ids", []), got.get("documents", [])) }
