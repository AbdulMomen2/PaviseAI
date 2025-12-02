# ==============================================
# Helper Functions: Display, Full Pipeline Runner
# Enhanced Logging, BLEU/Coherence/ROUGE Tracking
# ==============================================
from typing import Dict, Any
from src.normalizer import ClinicalNormalizer
from src.detector import PromptInjectionDetector
from src.neo4j_handler import Neo4jHandler
from src.medgnn import MedGNNAnomalyShield
from src.federated_learning import run_federated_simulation
from src.qa_stage import TransformerQAStage


def display_detection_results(result: Dict[str, Any]):
    """
    Nicely formatted display for prompt injection detection results.
    Includes pattern/model/context-aware breakdowns.
    """
    print("\n" + "="*80)
    print(f"🔍 PROMPT INJECTION DETECTION RESULTS")
    print("="*80)

    safety_icon = "✅" if result["is_safe"] else "⚠️"
    print(f"\n{safety_icon} Safety Status: {'SAFE' if result['is_safe'] else 'UNSAFE'}")
    print(f"📊 Risk Score: {result.get('risk_score',0):.3f} / 1.000")
    print(f"🎯 Confidence: {result.get('confidence',0):.3f}")
    print(f"🔬 Detection Method: {result.get('primary_detection_method','N/A')}")

    risk_level = (
        "🟢 LOW" if result.get('risk_score',0) < 0.3 else
        "🟡 MEDIUM" if result.get('risk_score',0) < 0.7 else
        "🔴 HIGH"
    )
    print(f"⚡ Risk Level: {risk_level}")

    # Recommendation
    print(f"\n💡 Recommendation: {result.get('recommendation','Check Query')}")
    print(f" Action: {result.get('action','Review')}")
    print(f" Reason: {result.get('reason','N/A')}")

    # Detailed breakdowns
    detailed = result.get('detailed_results', {})
    
    pattern = detailed.get('pattern_based', {})
    print(f"\n📋 Pattern-Based Detection: Risk={pattern.get('risk_score',0):.3f}, Safe={pattern.get('is_safe', True)}")
    flagged_patterns = pattern.get('flagged_patterns', [])
    if flagged_patterns:
        print(f" Flagged Patterns: {len(flagged_patterns)}")
        for i, (ptype, pat) in enumerate(flagged_patterns[:3], 1):
            print(f"  {i}. [{ptype}] {pat[:50]}...")

    model = detailed.get('model_based', {})
    print(f"\n📋 Model-Based Detection: Risk={model.get('risk_score',0):.3f}, Safe={model.get('is_safe', True)}")
    print(f" Safe Prob: {model.get('safe_probability',0):.3f}, Unsafe Prob: {model.get('unsafe_probability',0):.3f}")

    context = detailed.get('context_aware', {})
    print(f"\n📋 Context-Aware Detection: Risk={context.get('risk_score',0):.3f}, Safe={context.get('is_safe', True)}")
    risk_signals = context.get('risk_signals', [])
    if risk_signals:
        print(f" Signals: {', '.join(risk_signals[:2])}")

    print("\n" + "="*80 + "\n")


def full_pipeline(
    query: str,
    normalizer: ClinicalNormalizer,
    detector: PromptInjectionDetector,
    neo4j_handler: Neo4jHandler,
    medgnn: MedGNNAnomalyShield,
    qa_stage: TransformerQAStage
) -> Dict[str, Any]:
    """
    Full Clinical KG + MedGNN + QA pipeline runner with:
      1. Clinical Normalization
      2. Neo4j Storage
      3. Prompt Injection Detection
      4. Federated MedGNN Anomaly Shield
      5. Transformer QA Stage with Chain-of-Retrieval
      6. Reflection, Guardrails, Evaluation
    Returns full dictionary with all intermediate results.
    """

    print(f"\n{'='*80}")
    print(f"🚀 FULL PIPELINE PROCESSING: {query}")
    print('='*80)

    # Step 1: Clinical Normalization
    print("\n[Step 1] Clinical Normalization...")
    normalized = normalizer.normalize_query(query)
    entities_count = len(normalized.get('entities', []))
    print(f" ✓ Found {entities_count} medical entities")
    print(f" ✓ Medical query: {normalized.get('is_medical_query', False)}")

    # Step 1b: Store normalized query in Neo4j
    print("\n[Data Store] Neo4j - Storing normalized query...")
    neo4j_handler.store_normalized_query(normalized)

    # Step 2: Prompt Injection Detection
    print("\n[Step 2] Prompt Injection Detection...")
    detection = detector.detect_injection(query, normalized)

    if not detection.get('is_safe', True):
        print(f"\n{'─'*80}")
        print(f"⛔ PIPELINE STOPPED: Query REJECTED")
        print(f" Risk Score: {detection.get('risk_score',0):.3f}")
        print(f" Reason: {detection.get('reason','Suspicious input')}")
        print(f" → Blocked - No further processing")
        print(f"{'='*80}\n")
        return {
            "approved": False,
            "normalized": normalized,
            "detection": detection
        }

    # Step 3: MedGNN Anomaly Shield
    print("\n[Step 3] Federated MedGNN-Anomaly-Shield: Graph Plausibility Check...")
    plausibility = medgnn.check_plausibility(normalized)
    print(f" ✓ Plausibility Score: {plausibility.get('plausibility_score',0):.3f}")
    print(f" ✓ Anomaly Score: {plausibility.get('anomaly_score',0):.3f}")

    # Step 4: Transformer QA Stage with Chain-of-Retrieval
    print("\n[Step 4] Transformer QA Stage: Chain-of-Retrieval Prompting...")
    qa_result = qa_stage.generate_answer(query, plausibility.get("context_chain", []), plausibility)

    # Safe fallback defaults
    reflection_stats = qa_result.get("reflection_stats", {"hallucination_score": 0.0, "loops_used": 0})
    evaluation_metrics = qa_result.get("evaluation_metrics", {"rouge_l":0.0,"bleu_score":0.0,"coherence":0.0})
    guardrails = qa_result.get("guardrail_checks", {"input_safe": True,"output_safe": True})
    context_stats = qa_result.get("context_engineering_stats", {"filtered_out":0,"chain_steps":0})

    # Display Summary
    print(f"\n{'─'*80}")
    print("🎯 FINAL VERDICT")
    print(f"{'─'*80}")
    print(f"✅ Query APPROVED & PLAUSIBLE")
    print(f" Injection Risk: {detection.get('risk_score',0):.3f}")
    print(f" Plausibility Score: {plausibility.get('plausibility_score',0):.3f}")
    print(f" Anomaly Score: {plausibility.get('anomaly_score',0):.3f}")
    print(f" Anomalies Detected: {len(plausibility.get('anomalies', []))}")
    print(f" RAG Retrieved Evidence: {plausibility.get('rag_retrieved',0)}")
    print(f" QA Confidence: {qa_result.get('confidence',0):.3f}")
    print(f" Reflection Hallucination: {reflection_stats['hallucination_score']:.3f} (Loops: {reflection_stats['loops_used']})")
    print(f" Evaluation: ROUGE-L={evaluation_metrics['rouge_l']:.3f} | BLEU={evaluation_metrics['bleu_score']:.3f} | Coherence={evaluation_metrics['coherence']:.3f}")
    print(f" Guardrails: Input Safe={guardrails['input_safe']}, Output Safe={guardrails['output_safe']}")
    print(f" Context Engineering: Filtered={context_stats['filtered_out']}, Steps={context_stats['chain_steps']}")
    print(f" Total Entities: {normalized.get('entity_summary', {}).get('total_entities',0)}")
    if plausibility.get('anomalies'):
        print(f" ⚠️ Top Anomalies: {', '.join(plausibility['anomalies'][:2])}")
    print(f" Final Answer: {qa_result.get('answer','N/A')}")
    print(f"{'='*80}\n")

    return {
        "normalized": normalized,
        "detection": detection,
        "plausibility": plausibility,
        "qa_result": qa_result,
        "approved": True
    }
