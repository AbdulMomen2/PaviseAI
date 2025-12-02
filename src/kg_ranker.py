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
        # ✅ FIX: Use small embedding model to reduce RAM/paging memory issues
        self.model = SentenceTransformer(DENSE_EMBEDDING_MODEL)
        
        # Store triples
        self.triples = kg_triples

        # Encode triples to embeddings (float32 for FAISS)
        self.embeddings = np.array(self.model.encode(kg_triples, show_progress_bar=True), dtype='float32')

        # Initialize FAISS index (Inner Product = cosine similarity)
        self.index = faiss.IndexFlatIP(KG_TRIPLE_DIM)
        self.index.add(self.embeddings)
        print("✅ KG Ranker initialized (FAISS Vector DB)")

    def rank_triples(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Rank KG triples by similarity to query embedding."""
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype('float32')
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        return [
            {"triple": self.triples[i], "score": scores[0][j]}
            for j, i in enumerate(indices[0])
            if i != -1
        ]
