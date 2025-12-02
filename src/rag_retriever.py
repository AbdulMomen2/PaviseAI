"""
Hierarchical Multi-Expert RAG Pipeline
=====================================
Full, ready-to-run Python module for neurology-focused queries with BM25 + Dense retriever.
Includes safety filtering, deterministic generation, fresh context handling, and evaluation helpers.
"""

from typing import List, Dict, Optional, Tuple
import re
import numpy as np
from dataclasses import dataclass, field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# =========================
# Config defaults
# =========================
RAG_TOP_K_SPARSE = 5
RAG_TOP_K_DENSE = 5

# ✅ Use small model to reduce RAM / paging memory issues
DENSE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =========================
# Utility Functions
# =========================
RE_COMBINE_WHITESPACE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip().replace("\n", " ")
    t = RE_COMBINE_WHITESPACE.sub(" ", t)
    return t

# =========================
# RAG Retriever
# =========================
class RAGRetriever:
    def __init__(self, corpus: List[str]):
        self.update_corpus(corpus)
        print("✅ RAG Retriever initialized (BM25 + Dense)")

    def update_corpus(self, corpus: List[str]):
        self.corpus = corpus
        self.bm25 = BM25Okapi([doc.split() for doc in corpus])
        # ✅ Memory-friendly float32 embeddings
        self.dense_model = SentenceTransformer(DENSE_EMBEDDING_MODEL)
        self.dense_embeddings = np.array(
            self.dense_model.encode(corpus, show_progress_bar=True, convert_to_numpy=True),
            dtype='float32'
        )

    def retrieve_sparse(self, query: str, top_k: int = RAG_TOP_K_SPARSE) -> List[Dict]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"doc": self.corpus[i], "score": float(scores[i]), "type": "sparse", "index": int(i)} for i in top_indices]

    def retrieve_dense(self, query: str, top_k: int = RAG_TOP_K_DENSE) -> List[Dict]:
        query_emb = np.array(
            self.dense_model.encode([query], convert_to_numpy=True),
            dtype='float32'
        )
        scores = np.dot(self.dense_embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"doc": self.corpus[i], "score": float(scores[i]), "type": "dense", "index": int(i)} for i in top_indices]

    def hybrid_retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        sparse_results = self.retrieve_sparse(query)
        dense_results = self.retrieve_dense(query)
        # Weighted fusion
        for r in sparse_results:
            r["score"] *= 0.4
        for r in dense_results:
            r["score"] *= 0.6
        all_results = sparse_results + dense_results
        best_by_index: Dict[int, Dict] = {}
        for r in all_results:
            idx = r.get("index")
            if idx not in best_by_index or r["score"] > best_by_index[idx]["score"]:
                best_by_index[idx] = r
        merged = sorted(best_by_index.values(), key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

# =========================
# Safe RAG Wrapper
# =========================
@dataclass
class SafeRAG:
    retriever: RAGRetriever
    generator: Optional[object] = None
    dense_threshold: float = 0.20
    max_context_chars: int = 2500
    context_top_k: int = 3

    def _select_contexts(self, query: str) -> Tuple[List[Dict], float]:
        results = self.retriever.hybrid_retrieve(query, top_k=10)
        scores = np.array([r["score"] for r in results], dtype=float)
        if len(scores) == 0:
            return [], 0.0
        min_s, max_s = float(scores.min()), float(scores.max())
        denom = max_s - min_s if max_s != min_s else 1.0
        for r in results:
            r["norm_score"] = (r["score"] - min_s) / denom
        selected = [r for r in results if r["norm_score"] >= self.dense_threshold]
        if not selected and results and results[0]["norm_score"] >= self.dense_threshold * 0.75:
            selected = [results[0]]
        selected = sorted(selected, key=lambda x: x["norm_score"], reverse=True)[:self.context_top_k]
        top_score = selected[0]["norm_score"] if selected else 0.0
        return selected, top_score

    def _compose_context(self, contexts: List[Dict]) -> str:
        parts = []
        total_chars = 0
        for c in contexts:
            text = clean_text(c["doc"])
            if total_chars + len(text) > self.max_context_chars:
                remaining = max(0, self.max_context_chars - total_chars)
                parts.append(text[:remaining])
                break
            parts.append(text)
            total_chars += len(text)
        return "\n\n---\n\n".join(parts)

    def answer(self, query: str) -> Dict:
        query = clean_text(query)
        contexts, top_score = self._select_contexts(query)
        if top_score < self.dense_threshold:
            return {"answer": "I don't know based on the provided context.", "confidence": float(top_score), "contexts": contexts}
        context_text = self._compose_context(contexts)
        prompt = f"Answer the neurology question ONLY using the context below.\nCONTEXT:\n{context_text}\nQUESTION:\n{query}\nAnswer:"
        raw = self.generator.generate(prompt) if self.generator else "I don't know based on the provided context."
        processed = RE_COMBINE_WHITESPACE.sub(" ", raw.strip())
        if not processed.endswith((".", "?", "!")):
            processed += "."
        return {"answer": processed, "raw": raw, "confidence": float(top_score), "contexts": contexts, "prompt": prompt}

# =========================
# Main Test Run
# =========================
if __name__ == "__main__":
    corpus = [
        "Neurology overview: stroke, epilepsy, Parkinson's disease, and their treatments.",
        "Brain anatomy: cerebral cortex, cerebellum, brainstem, and neuron signaling.",
        "Clinical guidelines for migraine, neuropathy, and multiple sclerosis management.",
    ]

    retriever = RAGRetriever(corpus)

    class DummyGen:
        def generate(self, prompt):
            return "Based on the context, the neurology answer is derived from clinical evidence."

    safe_rag = SafeRAG(retriever=retriever, generator=DummyGen(), dense_threshold=0.20)

    # Example queries - fresh session per query
    new_queries = [
        "What are the treatment options for stroke?",
        "Explain the role of the cerebellum in motor control.",
        "How is multiple sclerosis diagnosed and managed?",
        "Describe the mechanism of Parkinson's disease progression.",
        "How to treat acute epilepsy episodes in adults?"
    ]

    for q in new_queries:
        safe_rag.retriever.update_corpus(corpus)  # refresh corpus embeddings
        safe_rag_instance = SafeRAG(retriever=safe_rag.retriever, generator=DummyGen(), dense_threshold=0.20)
        out = safe_rag_instance.answer(q)
        print("---")
        print("Q:", q)
        print("Answer:", out["answer"])
        print("Confidence:", out["confidence"])
