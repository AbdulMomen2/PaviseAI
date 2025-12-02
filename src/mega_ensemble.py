# Mega-RAG Ensemble: Weighted Fusion of Sparse, Dense, and KG Retrievals
# =====================================================
from typing import List, Dict
from src.config import RAG_ENSEMBLE_ALPHA  # Weight for sparse
import logging

logger = logging.getLogger(__name__)


class MegaRAGEnsemble:
    """
    Ensemble module: combines retrievals from sparse, dense, and KG ranking.
    Uses weighted reciprocal rank fusion (RRF) to generate a final ranked list.
    Outputs: list of docs/triples with ensemble_score for downstream QA.
    """

    def __init__(self):
        logger.info("✅ Mega-RAG Ensemble initialized")

    def ensemble(
        self,
        sparse_results: List[Dict],
        dense_results: List[Dict],
        kg_results: List[Dict],
        alpha: float = RAG_ENSEMBLE_ALPHA
    ) -> List[Dict]:
        """
        Combine results using weighted scoring.

        Args:
            sparse_results: List of dicts from sparse retrieval (each dict must have 'doc' and 'score')
            dense_results: List of dicts from dense retrieval
            kg_results: List of dicts from KG ranking
            alpha: weight for sparse results (0 < alpha < 1)

        Returns:
            ranked: List[Dict] sorted by combined ensemble score
        """
        all_results = {}

        # Internal helper: safely add results to all_results dict
        def add_result(res_list: List[Dict], source: str):
            for res in res_list:
                # Safe key retrieval
                doc_content = (
                    res.get("doc") or
                    res.get("text") or
                    res.get("summary") or
                    str(res)
                )
                try:
                    score = float(res.get("score", 0.0))
                except (ValueError, TypeError):
                    score = 0.0
                doc_id = hash(doc_content)

                if doc_id not in all_results:
                    all_results[doc_id] = {
                        "doc": doc_content,
                        "scores": {"sparse": [], "dense": [], "kg": []},
                        "ensemble_score": 0.0
                    }

                all_results[doc_id]["scores"][source].append(score)

        # Add all retrieval results
        add_result(sparse_results, "sparse")
        add_result(dense_results, "dense")
        add_result(kg_results, "kg")

        # Compute weighted ensemble score for each document
        for doc_id, info in all_results.items():
            sparse_score = sum(info["scores"]["sparse"]) * alpha
            dense_score = sum(info["scores"]["dense"]) * (1 - alpha) / 2
            kg_score = sum(info["scores"]["kg"]) * (1 - alpha) / 2

            # Combined ensemble score
            ensemble_score = sparse_score + dense_score + kg_score

            # Clamp score between 0 and 1 for stability
            info["ensemble_score"] = max(0.0, min(ensemble_score, 1.0))

        # Sort results by ensemble_score descending
        ranked = sorted(all_results.values(), key=lambda x: x["ensemble_score"], reverse=True)

        # Optional: limit output to top K (default = number of sparse_results)
        top_k = len(sparse_results) if sparse_results else len(ranked)
        logger.info(f"📊 Mega-RAG Ensemble: Returning top {top_k} results")

        return ranked[:top_k]
