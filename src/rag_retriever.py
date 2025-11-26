# Hierarchical Multi-Expert RAG: Sparse (BM25) + Dense (CPT-like Embeddings) Retrieval
# =====================================================
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
from src.config import RAG_TOP_K_SPARSE, RAG_TOP_K_DENSE, DENSE_EMBEDDING_MODEL

class RAGRetriever:
    """
    Hybrid Retriever: BM25 (Sparse) + Dense Embeddings (e.g., MiniLM for CPT-like).
    Retrieves from corpus (e.g., PubMed abstracts or KG texts).
    """
    def __init__(self, corpus: List[str]):
        self.corpus = corpus  # List of documents (e.g., PubMed summaries + KG triples)
        self.bm25 = BM25Okapi([doc.split() for doc in corpus])
        self.dense_model = SentenceTransformer(DENSE_EMBEDDING_MODEL)
        self.dense_embeddings = self.dense_model.encode(corpus)
        print("✅ RAG Retriever initialized (BM25 + Dense)")

    def retrieve_sparse(self, query: str, top_k: int = RAG_TOP_K_SPARSE) -> List[Dict]:
        """BM25 Sparse Retrieval."""
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"doc": self.corpus[i], "score": scores[i], "type": "sparse"} for i in top_indices]

    def retrieve_dense(self, query: str, top_k: int = RAG_TOP_K_DENSE) -> List[Dict]:
        """Dense Retrieval with Embeddings."""
        query_emb = self.dense_model.encode([query])
        scores = np.dot(self.dense_embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"doc": self.corpus[i], "score": scores[i], "type": "dense"} for i in top_indices]

    def hybrid_retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Combine Sparse + Dense."""
        sparse_results = self.retrieve_sparse(query)
        dense_results = self.retrieve_dense(query)
        # Simple reciprocal rank fusion (RRf)
        all_results = sparse_results + dense_results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]