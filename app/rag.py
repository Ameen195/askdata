"""Local BM25, FAISS, and cross-encoder retrieval for policy documents."""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
DOCS_DIR = ROOT / "data" / "docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_embedder: Any | None = None
_reranker: Any | None = None


def _models() -> Any:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _token_chunks(text: str, tokenizer: Any, size: int = 500, overlap: int = 80) -> list[str]:
    """Split text by model tokens, retaining the requested overlap."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    stride = size - overlap
    chunks = []
    start = 0
    while start < len(ids):
        chunk = tokenizer.decode(ids[start:start + size], skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(ids):
            break
        start += stride
    return chunks


def _documents(path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(path)
    files = sorted(source_path.rglob("*") if source_path.is_dir() else [source_path])
    docs = []
    for file in files:
        if file.suffix.lower() == ".md":
            docs.append((file, file.read_text(encoding="utf-8"), 1))
        elif file.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            for page_no, page in enumerate(PdfReader(str(file)).pages, start=1):
                docs.append((file, page.extract_text() or "", page_no))
    tokenizer = _models().tokenizer
    records = []
    for file, text, page in docs:
        for chunk in _token_chunks(text, tokenizer):
            records.append({"text": chunk, "source": file.name, "page": page})
    return records


def ingest(path: str | Path) -> int:
    """Ingest Markdown/PDF files and persist vector and BM25 indexes."""
    records = _documents(path)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError(f"No readable PDF or Markdown documents found in {path}")
    vectors = _models().encode([d["text"] for d in records], normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR / "vectors.faiss"))
    (INDEX_DIR / "documents.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    bm25 = BM25Okapi([_tokenize(d["text"]) for d in records])
    with (INDEX_DIR / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)
    return len(records)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _load_index() -> tuple[list[dict[str, Any]], Any, BM25Okapi]:
    docs_path = INDEX_DIR / "documents.json"
    vec_path = INDEX_DIR / "vectors.faiss"
    bm_path = INDEX_DIR / "bm25.pkl"
    if not (docs_path.exists() and vec_path.exists() and bm_path.exists()):
        if DOCS_DIR.exists() and any(DOCS_DIR.rglob("*")):
            ingest(DOCS_DIR)
        else:
            return [], None, BM25Okapi([[]])
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    import faiss

    index = faiss.read_index(str(vec_path))
    with bm_path.open("rb") as f:
        bm25 = pickle.load(f)
    return docs, index, bm25


def retrieve(query: str, k: int = 5, mode: str = "hybrid_rerank") -> list[dict[str, Any]]:
    """Retrieve with bm25, vector, hybrid RRF, or hybrid plus cross-encoder."""
    if mode not in {"bm25", "vector", "hybrid", "hybrid_rerank"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    docs, index, bm25 = _load_index()
    if not docs:
        return []
    n = len(docs)
    b_scores = np.asarray(bm25.get_scores(_tokenize(query)))
    b_order = np.argsort(b_scores)[::-1][:min(20, n)].tolist()
    qvec = _models().encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    v_scores, v_ids = index.search(qvec, min(20, n))
    v_order = [int(i) for i in v_ids[0] if i >= 0]
    if mode == "bm25":
        selected = b_order[:k]
        scores = {i: float(b_scores[i]) for i in selected}
    elif mode == "vector":
        selected = v_order[:k]
        scores = {int(i): float(s) for i, s in zip(v_ids[0], v_scores[0]) if i >= 0}
    else:
        fused: dict[int, float] = {}
        for order in (b_order, v_order):
            for rank, idx in enumerate(order, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank)
        selected = sorted(fused, key=fused.get, reverse=True)[:20]
        scores = fused
        if mode == "hybrid_rerank":
            global _reranker
            if _reranker is None:
                from sentence_transformers import CrossEncoder

                _reranker = CrossEncoder(RERANK_MODEL)
            pairs = [(query, docs[i]["text"]) for i in selected]
            rank_scores = np.asarray(_reranker.predict(pairs)).reshape(-1)
            ranked = sorted(zip(selected, rank_scores), key=lambda x: float(x[1]), reverse=True)
            selected = [i for i, _ in ranked[:k]]
            scores = {i: float(s) for i, s in ranked}
        else:
            selected = selected[:k]
    return [{**docs[i], "score": float(scores.get(i, 0.0))} for i in selected]


def hybrid_retrieve(query: str, k: int = 5, mode: str = "hybrid_rerank") -> list[dict[str, Any]]:
    """Compatibility entry point used by the graph and retrieval evaluations."""
    return retrieve(query, k=k, mode=mode)
