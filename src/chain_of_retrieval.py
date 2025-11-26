# Chain-of-Retrieval Prompting with Advanced Context Engineering (Modular)
# =====================================================
import numpy as np
from typing import List, Dict   # ❗ FIXED (removed `str`)
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
        print("✅ Chain-of-Retrieval Module initialized with Context Engineering")

    def engineer_context_chain(self, context_chain: List[Dict], threshold: float = CONTEXT_RELEVANCE_THRESHOLD) -> str:
        """Engineer Context: Filter, Structure, and Chain."""
        # Filter low-relevance
        filtered = [ctx for ctx in context_chain if ctx.get('score', 0) > threshold]
        
        # Categorize & Structure (Chain Steps)
        entities = [ctx for ctx in filtered if 'Entity' in str(ctx.get('doc', '')) or 'CUI' in str(ctx.get('doc', ''))]
        evidence = [ctx for ctx in filtered if 'PubMed' in str(ctx.get('doc', ''))]
        kg_rels = [ctx for ctx in filtered if 'triple' in str(ctx) or 'KG' in str(ctx.get('doc', ''))]
        
        chain_parts = []
        if entities:
            avg_score = np.mean([e['score'] for e in entities])
            chain_parts.append(
                f"Step 1: UMLS Entities - "
                f"{', '.join([e['doc'][:50] + '...' for e in entities[:2]])} "
                f"(Avg Relevance: {avg_score:.2f})"
            )
        if evidence:
            avg_score = np.mean([ev['score'] for ev in evidence])
            chain_parts.append(
                f"Step 2: PubMed Evidence - "
                f"{', '.join([ev['doc'][:50] + '...' for ev in evidence[:2]])} "
                f"(Avg Relevance: {avg_score:.2f})"
            )
        if kg_rels:
            avg_score = np.mean([rel['score'] for rel in kg_rels])
            chain_parts.append(
                f"Step 3: KG Relations - "
                f"{', '.join([rel['triple'][:50] + '...' for rel in kg_rels[:2]])} "
                f"(Avg Relevance: {avg_score:.2f})"
            )
        
        return "\n".join(chain_parts) if chain_parts else "No high-relevance context available. Rely on general knowledge."

    def inject_few_shot_examples(self) -> str:
        """Inject Few-Shot for Consistency."""
        examples = "\n".join([
            f"Example Query: {ex['query']}\n"
            f"Example Chain: {ex['context_chain']}\n"
            f"Example Answer: {ex['answer']}\n---"
            for ex in FEW_SHOT_EXAMPLES
        ])
        return examples

    def build_engineered_prompt(self, query: str, engineered_chain: str, anomaly_note: str = "") -> str:
        """Build Final Prompt with Role-Playing & Structure."""
        few_shot = self.inject_few_shot_examples()
        anomaly_suffix = f"\n[Anomaly Alert]: {anomaly_note} - Prioritize verified facts." if anomaly_note else ""
        
        prompt = CHAIN_OF_RETRIEVAL_PROMPT.format(
            examples=few_shot,
            threshold=CONTEXT_RELEVANCE_THRESHOLD,
            context_chain=engineered_chain + anomaly_suffix,
            query=query
        )
        return prompt

    def process_context_for_qa(self, context_chain: List[Dict], query: str, anomalies: List[str]) -> Dict:
        """Full Processing: Engineer → Build Prompt → Return for LLM."""
        engineered_chain = self.engineer_context_chain(context_chain)
        anomaly_note = ", ".join(anomalies[:2]) if anomalies else ""
        prompt = self.build_engineered_prompt(query, engineered_chain, anomaly_note)
        
        print(f"🔧 Engineered Chain Preview: {engineered_chain[:150]}...")
        return {
            "engineered_prompt": prompt,
            "filtered_contexts": len(context_chain) - len(
                [ctx for ctx in context_chain if ctx.get('score', 0) > CONTEXT_RELEVANCE_THRESHOLD]
            ),
            "chain_length": len(engineered_chain.split('\n'))
        }
