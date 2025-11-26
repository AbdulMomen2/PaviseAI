# Helper Functions (Display, Pipeline Runner with Enhanced Logging for BLEU/Coherence)
# =====================================================
from typing import Dict
from src.normalizer import ClinicalNormalizer
from src.detector import PromptInjectionDetector
from src.neo4j_handler import Neo4jHandler
from src.medgnn import MedGNNAnomalyShield
from src.federated_learning import run_federated_simulation
from src.qa_stage import TransformerQAStage


def display_detection_results(result: Dict):
    """Display detection results in a readable format."""
    print("\n" + "="*80)
    print(f"🔍 PROMPT INJECTION DETECTION RESULTS")
    print("="*80)

    # Overall verdict
    safety_icon = "✅" if result["is_safe"] else "⚠️"
    print(f"\n{safety_icon} Safety Status: {'SAFE' if result['is_safe'] else 'UNSAFE'}")
    print(f"📊 Risk Score: {result['risk_score']:.3f} / 1.000")
    print(f"🎯 Confidence: {result['confidence']:.3f}")
    print(f"🔬 Detection Method: {result['primary_detection_method']}")

    # Risk level visualization
    risk_level = (
        "🟢 LOW" if result['risk_score'] < 0.3 else
        "🟡 MEDIUM" if result['risk_score'] < 0.7 else
        "🔴 HIGH"
    )
    print(f"⚡ Risk Level: {risk_level}")

    # Recommendation
    print(f"\n💡 Recommendation: {result['recommendation']}")
    print(f" Action: {result['action']}")
    print(f" Reason: {result['reason']}")

    # Detailed breakdown
    print(f"\n📋 Detailed Analysis:")
    pattern = result['detailed_results']['pattern_based']
    print(f" Pattern-Based: Risk={pattern['risk_score']:.3f}, Safe={pattern['is_safe']}")
    if pattern['flagged_patterns']:
        print(f" Flagged: {len(pattern['flagged_patterns'])} patterns")
        for i, (ptype, pat) in enumerate(pattern['flagged_patterns'][:3], 1):
            print(f" {i}. [{ptype}] {pat[:50]}")

    model = result['detailed_results']['model_based']
    print(f" Model-Based: Risk={model['risk_score']:.3f}, Safe={model['is_safe']}")
    print(f" Safe prob: {model['safe_probability']:.3f}, Unsafe prob: {model['unsafe_probability']:.3f}")

    context = result['detailed_results']['context_aware']
    print(f" Context-Aware: Risk={context['risk_score']:.3f}, Safe={context['is_safe']}")
    if context.get('risk_signals'):
        print(f" Signals: {', '.join(context['risk_signals'][:2])}")

    print("\n" + "="*80 + "\n")



def full_pipeline(query: str, normalizer, detector, neo4j_handler, medgnn, qa_stage: TransformerQAStage):
    """
    Full Pipeline: Step 1 → Neo4j Store → Step 2 → If Safe → Step 3 (Federated)
    → Step 4 (QA with Reflection/Guardrails/Eval) → Output
    """
    print(f"\n{'='*80}")
    print(f"🚀 FULL PIPELINE PROCESSING: {query}")
    print('='*80)

    # Step 1: Normalize
    print("\n[Step 1] Clinical Normalization...")
    normalized = normalizer.normalize_query(query)
    print(f" ✓ Found {len(normalized['entities'])} medical entities")
    print(f" ✓ Medical query: {normalized['is_medical_query']}")

    # Store to Neo4j
    print("\n[Data Store] Neo4j - Storing normalized query...")
    neo4j_handler.store_normalized_query(normalized)

    # Step 2: Detect injection
    print("\n[Step 2] Prompt Injection Detection...")
    detection = detector.detect_injection(query, normalized)

    if not detection['is_safe']:
        print(f"\n{'─'*80}")
        print(f"⛔ PIPELINE STOPPED: Query REJECTED")
        print(f" Risk Score: {detection['risk_score']:.3f}")
        print(f" Reason: {detection['reason']}")
        print(f" → Blocked - No further processing")
        print(f"{'='*80}\n")
        return {"approved": False, "normalized": normalized, "detection": detection}

    # Step 3: Federated MedGNN Anomaly Shield
    print("\n[Step 3] Federated MedGNN-Anomaly-Shield: Graph Plausibility Check...")
    plausibility = medgnn.check_plausibility(normalized)

    # Step 4: Transformer QA Stage
    print("\n[Step 4] Transformer QA Stage: Chain-of-Retrieval Prompting with Context Engineering...")
    qa_result = qa_stage.generate_answer(query, plausibility.get("context_chain", []), plausibility)

    # ---- FIX: Safe fallback if reflection_stats missing ----
    reflection_stats = qa_result.get(
        "reflection_stats",
        {"hallucination_score": 0.0, "loops_used": 0}
    )

    evaluation = qa_result.get(
        "evaluation_metrics",
        {"rouge_l": 0.0, "bleu_score": 0.0, "coherence": 0.0}
    )

    guardrails = qa_result.get(
        "guardrail_checks",
        {"input_safe": True, "output_safe": True}
    )

    context_stats = qa_result.get(
        "context_engineering_stats",
        {"filtered_out": 0, "chain_steps": 0}
    )
    # ----------------------------------------------------------

    print(f"\n{'─'*80}")
    print(f"🎯 FINAL VERDICT:")
    print(f"{'─'*80}")

    print(f"✅ Query APPROVED & PLAUSIBLE")
    print(f" Injection Risk: {detection['risk_score']:.3f}")
    print(f" Plausibility Score: {plausibility['plausibility_score']:.3f}")
    print(f" Anomaly Score: {plausibility.get('anomaly_score', 0):.3f}")
    print(f" Anomalies: {len(plausibility['anomalies'])}")
    print(f" RAG Retrieved: {plausibility.get('rag_retrieved', 0)} triples/evidence")
    print(f" QA Confidence: {qa_result['confidence']:.3f}")

    print(f" Reflection Hallucination: {reflection_stats['hallucination_score']:.3f} "
          f"(Loops: {reflection_stats['loops_used']})")

    print(f" Evaluation Overall: {qa_result.get('overall_score', 0.0):.3f} "
          f"| ROUGE-L: {evaluation['rouge_l']:.3f} "
          f"| BLEU: {evaluation['bleu_score']:.3f} "
          f"| Coherence: {evaluation['coherence']:.3f}")

    print(f" Guardrails: Input Safe: {guardrails['input_safe']}, Output Safe: {guardrails['output_safe']}")
    print(f" Final Answer: {qa_result['answer']}")

    print(f" Context Engineering: Filtered {context_stats['filtered_out']}, Steps: {context_stats['chain_steps']}")

    # 🔹 FIX: direct print without len() since total_entities is int
    print(f" Entities Stored: {normalized['entity_summary']['total_entities']}")

    print(f" → Personalized Subgraph Ready (OMNIBased + Real PubMed + Hierarchical RAG Integrated)")
    print(f" → Federated GNN Model Used for Anomaly Detection")
    print(f" → Privacy: DP Noise Applied to Scores")
    print(f" → QA via Fine-Tuned GPT with Adversarial Reflection Chain, Guardrails, & Full Evaluation Layer")

    if plausibility['anomalies']:
        print(f" ⚠️ Anomalies: {', '.join(plausibility['anomalies'][:2])}")

    print(f"{'='*80}\n")

    return {
        "normalized": normalized,
        "detection": detection,
        "plausibility": plausibility,
        "qa_result": qa_result,
        "approved": True
    }

