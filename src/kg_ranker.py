# KG Triple Vector Ranking or Database
# =====================================================
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from src.config import KG_TRIPLE_DIM, DENSE_EMBEDDING_MODEL

class KGRanker:
    """
    Vector Ranking for KG Triples (e.g., (Entity, Relation, Entity)).
    Uses FAISS for efficient similarity search.
    """
    def __init__(self, kg_triples: List[str]):  # Triples as strings: "Disease-TREATS-Drug"
        self.model = SentenceTransformer(DENSE_EMBEDDING_MODEL)
        self.triples = kg_triples
        self.embeddings = self.model.encode(kg_triples)
        self.index = faiss.IndexFlatIP(KG_TRIPLE_DIM)  # Inner product for cosine sim
        self.index.add(np.array(self.embeddings).astype('float32'))
        print("✅ KG Ranker initialized (FAISS Vector DB)")

    def rank_triples(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Rank KG triples by similarity to query embedding."""
        scores, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), top_k)
        return [{"triple": self.triples[i], "score": scores[0][j]} for j, i in enumerate(indices[0]) if i != -1]