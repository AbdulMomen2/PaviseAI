# Chain-of-Retrieval Prompting with Advanced Context Engineering (Modular)
# =====================================================
import re
from typing import List, Dict, Any, Optional
import numpy as np
from src.config import CONTEXT_RELEVANCE_THRESHOLD, FEW_SHOT_EXAMPLES, CHAIN_OF_RETRIEVAL_PROMPT


class ChainOfRetrieval:
    """
    Dedicated Module: Chain-of-Retrieval Prompting with Context Engineering.
    - Dynamic filtering: Relevance-based pruning.
    - Structured chaining: Entities → Evidence → KG Relations.
    - Few-shot injection: Consistency via examples.
    - Role-playing: Expert persona.
    - Output: Engineered prompt ready for LLM.
    """

    def __init__(self):
        # kept simple print to preserve current behavior; replace with logger if desired
        print("✅ Chain-of-Retrieval Module initialized with Context Engineering")

    @staticmethod
    def _safe_snippet(text: Any, length: int = 50) -> str:
        """
        Return a safe string snippet of given length. Handles non-string inputs.
        """
        if text is None:
            return ""
        s = str(text)
        s = s.strip().replace("\n", " ")
        if len(s) <= length:
            return s
        return s[:length] + "..."

    @staticmethod
    def _mean_score(items: List[Dict[str, Any]], key: str = "score") -> float:
        """
        Compute mean safely. Return 0.0 if items is empty or have no score.
        """
        scores = [float(it.get(key, 0.0)) for it in items if it is not None and key in it]
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Basic normalization to improve lexical overlap for evaluation/comparison.
        Kept minimal — you can extend with language-specific rules.
        """
        t = text.lower().strip()
        t = re.sub(r"\s+", " ", t)
        return t

    def engineer_context_chain(
        self,
        context_chain: List[Dict[str, Any]],
        threshold: float = CONTEXT_RELEVANCE_THRESHOLD
    ) -> str:
        """Engineer Context: Filter, Structure, and Chain."""

        # Guard: ensure context_chain is list
        if not isinstance(context_chain, list):
            raise ValueError("context_chain must be a List[Dict].")

        # Filter high-relevance contexts (strict greater than threshold)
        filtered = [ctx for ctx in context_chain if float(ctx.get("score", 0.0)) > float(threshold)]

        # Categorize & Structure (Chain Steps)
        entities = [
            ctx for ctx in filtered
            if "Entity" in str(ctx.get("doc", "")) or "CUI" in str(ctx.get("doc", ""))
        ]
        evidence = [
            ctx for ctx in filtered
            if "PubMed" in str(ctx.get("doc", "")) or "pubmed" in str(ctx.get("doc", ""))
        ]
        kg_rels = [
            ctx for ctx in filtered
            if ("triple" in ctx) or ("KG" in str(ctx.get("doc", ""))) or ("kg" in str(ctx.get("doc", "")))
        ]

        chain_parts: List[str] = []

        if entities:
            avg_score = self._mean_score(entities, key="score")
            snippets = ", ".join([self._safe_snippet(e.get("doc", ""), 50) for e in entities[:2]])
            chain_parts.append(
                f"Step 1: UMLS Entities - {snippets} (Avg Relevance: {avg_score:.2f})"
            )

        if evidence:
            avg_score = self._mean_score(evidence, key="score")
            snippets = ", ".join([self._safe_snippet(ev.get("doc", ""), 50) for ev in evidence[:2]])
            chain_parts.append(
                f"Step 2: PubMed Evidence - {snippets} (Avg Relevance: {avg_score:.2f})"
            )

        if kg_rels:
            avg_score = self._mean_score(kg_rels, key="score")
            # triple might be nested or in 'doc'; guard for both
            def triple_snippet(rel: Dict[str, Any]) -> str:
                if "triple" in rel:
                    return self._safe_snippet(rel.get("triple", ""), 50)
                # fallback: try to extract a triple-like string from doc
                return self._safe_snippet(rel.get("doc", ""), 50)
            snippets = ", ".join([triple_snippet(rel) for rel in kg_rels[:2]])
            chain_parts.append(
                f"Step 3: KG Relations - {snippets} (Avg Relevance: {avg_score:.2f})"
            )

        if chain_parts:
            return "\n".join(chain_parts)
        # If nothing passed the threshold, return explicit fallback message
        return "No high-relevance context available. Rely on general knowledge."

    def inject_few_shot_examples(self) -> str:
        """Inject Few-Shot for Consistency."""
        if not FEW_SHOT_EXAMPLES:
            return ""

        examples_strs = []
        for ex in FEW_SHOT_EXAMPLES:
            # Safely extract fields with fallbacks to avoid KeyError
            q = ex.get("query", "")
            ctx = ex.get("context_chain", "")
            ans = ex.get("answer", "")
            examples_strs.append(
                f"Example Query: {q}\nExample Chain: {ctx}\nExample Answer: {ans}\n---"
            )
        return "\n".join(examples_strs)

    def build_engineered_prompt(self, query: str, engineered_chain: str, anomaly_note: Optional[str] = "") -> str:
        """Build Final Prompt with Role-Playing & Structure."""
        few_shot = self.inject_few_shot_examples()
        anomaly_suffix = f"\n[Anomaly Alert]: {anomaly_note} - Prioritize verified facts." if anomaly_note else ""

        # Ensure prompt template has expected placeholders; if not, try to format safely
        try:
            prompt = CHAIN_OF_RETRIEVAL_PROMPT.format(
                examples=few_shot,
                threshold=CONTEXT_RELEVANCE_THRESHOLD,
                context_chain=engineered_chain + anomaly_suffix,
                query=query
            )
        except Exception as e:
            # Fallback: build a simple composed prompt to avoid complete failure
            print(f"⚠️ Prompt template formatting failed: {e}. Using fallback prompt assembly.")
            prompt_parts = [
                "Role: Expert assistant specialized in retrieval and evidence-based answers.",
                f"Threshold: {CONTEXT_RELEVANCE_THRESHOLD}",
                "Few-shot examples:",
                few_shot or "(none)",
                "Engineered Context Chain:",
                engineered_chain + anomaly_suffix,
                "User Query:",
                query
            ]
            prompt = "\n\n".join(prompt_parts)

        return prompt

    def process_context_for_qa(
        self,
        context_chain: List[Dict[str, Any]],
        query: str,
        anomalies: Optional[List[str]] = None,
        threshold: float = CONTEXT_RELEVANCE_THRESHOLD
    ) -> Dict[str, Any]:
        """Full Processing: Engineer → Build Prompt → Return for LLM."""
        if anomalies is None:
            anomalies = []

        engineered_chain = self.engineer_context_chain(context_chain, threshold=threshold)
        anomaly_note = ", ".join(anomalies[:2]) if anomalies else ""
        prompt = self.build_engineered_prompt(query, engineered_chain, anomaly_note)

        # Number of contexts removed (filtered out) vs retained
        retained = [ctx for ctx in context_chain if float(ctx.get("score", 0.0)) > float(threshold)]
        filtered_out_count = len(context_chain) - len(retained)

        # chain_length as number of steps (lines) in engineered_chain, treat fallback as 0
        chain_lines = engineered_chain.split("\n") if engineered_chain and "No high-relevance" not in engineered_chain else []
        chain_length = len(chain_lines)

        # Debug/preview (kept print to match original behavior)
        preview = engineered_chain if len(engineered_chain) <= 150 else engineered_chain[:150] + "..."
        print(f"🔧 Engineered Chain Preview: {preview}")

        return {
            "engineered_prompt": prompt,
            "filtered_contexts": filtered_out_count,
            "chain_length": chain_length,
            "retained_contexts": len(retained),
            "engineered_chain": engineered_chain
        }
