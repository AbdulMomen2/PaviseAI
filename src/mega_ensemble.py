# Mega-RAG Ensemble: Combine Retrievals from Sparse, Dense, KG Ranking
# =====================================================
from typing import List, Dict
from src.config import RAG_ENSEMBLE_ALPHA  # Weight for sparse

class MegaRAGEnsemble:
    """
    Ensemble: Weighted fusion of Sparse, Dense, and KG results.
    Outputs ranked list of relevant docs/triples.
    """
    def __init__(self):
        print("✅ Mega-RAG Ensemble initialized")

    def ensemble(
        self,
        sparse_results: List[Dict],
        dense_results: List[Dict],
        kg_results: List[Dict],
        alpha: float = RAG_ENSEMBLE_ALPHA
    ) -> List[Dict]:
        """Reciprocal Rank Fusion (RRF) with weights and safe key handling."""
        
        all_results = {}

        # Helper to assign score with source label
        def add_result(res_list: List[Dict], source: str):
            for res in res_list:
                doc_content = res.get("doc") or res.get("text") or res.get("summary") or str(res)
                score = res.get("score", 0.0)
                doc_id = hash(doc_content)
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        "doc": doc_content,
                        "scores": {"sparse": [], "dense": [], "kg": []}
                    }
                all_results[doc_id]["scores"][source].append(score)

        # Add results safely
        add_result(sparse_results, "sparse")
        add_result(dense_results, "dense")
        add_result(kg_results, "kg")

        # Compute weighted ensemble score
        for doc_id, info in all_results.items():
            sparse_score = sum(info["scores"]["sparse"]) * alpha
            dense_score = sum(info["scores"]["dense"]) * (1 - alpha) / 2
            kg_score = sum(info["scores"]["kg"]) * (1 - alpha) / 2
            info["ensemble_score"] = sparse_score + dense_score + kg_score

        # Sort by ensemble score
        ranked = sorted(all_results.values(), key=lambda x: x["ensemble_score"], reverse=True)
        return ranked[:len(sparse_results)]  # Top K
