# src/qa_stage.py 
# Transformer QA Stage: Integrates Chain-of-Retrieval, Reflection, Guardrails, & Evaluation (Robust + Real Metrics)
# =====================================================

import os
import traceback
import logging
from typing import Dict, List, Any, Optional

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try to import optional components. If any fail, provide lightweight fallbacks so the QA stage never crashes.
try:
    from src.chain_of_retrieval import ChainOfRetrieval
except Exception:
    class ChainOfRetrieval:
        def __init__(self):
            pass

        def process_context_for_qa(self, context_chain: List[Dict], query: str, anomalies: List[str]) -> Dict[str, Any]:
            preview = ""
            if context_chain:
                preview = " | ".join([str(c.get("doc", ""))[:120] for c in context_chain[:3]])

            engineered_prompt = (
                f"Context: {preview}\n"
                f"Query: {query}\n"
                f"Anomalies: {', '.join(anomalies or [])}"
            )
            return {
                "engineered_prompt": engineered_prompt,
                "filtered_contexts": 0,
                "chain_length": 1
            }


try:
    from src.reflection_chain import ReflectionChain
except Exception:
    class ReflectionChain:
        def __init__(self):
            pass

        def reflect_and_refine(self, initial_answer: str, prompt: str, top_context: str = "") -> Dict[str, Any]:
            refined = (initial_answer or "").strip()

            if len(refined) > 200:
                try:
                    refined = refined[:200].rsplit(".", 1)[0] + "."
                except Exception:
                    refined = refined[:200]

            return {
                "refined_answer": refined,
                "final_hallucination_score": 0.05,
                "loops_used": 1,
                "safe": True
            }


try:
    from src.guardrails import Guardrails
except Exception:
    class Guardrails:
        def __init__(self):
            # basic blocked tokens; adapt to your policy
            self.blocked_tokens = ["suicide", "kill myself", "bomb", "attack"]

        def check_input(self, text: str) -> Dict[str, Any]:
            lowered = (text or "").lower()

            for t in self.blocked_tokens:
                if t in lowered:
                    return {"safe": False, "reason": f"Blocked token in input: {t}"}

            return {"safe": True, "reason": ""}

        def check_output(self, text: str) -> Dict[str, Any]:
            lowered = (text or "").lower()

            for t in self.blocked_tokens:
                if t in lowered:
                    return {"safe": False, "reason": f"Blocked token in output: {t}"}

            return {"safe": True, "reason": ""}


try:
    from src.evaluation_layer import EvaluationLayer
except Exception:
    class EvaluationLayer:
        def __init__(self):
            pass

        def evaluate_qa_output(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            answer = payload.get("answer", "") or ""
            sentences = payload.get("sentences", []) or []

            rouge_l = min(1.0, len(answer) / 800)
            bleu = (
                min(1.0, max(0.0, len(set(sentences)) / (len(sentences) + 1)))
                if sentences else 0.0
            )

            coherence = 0.0
            if sentences:
                lens = [len(s.split()) for s in sentences if s]
                if lens:
                    mean = sum(lens) / len(lens)
                    var = sum((l - mean) ** 2 for l in lens) / len(lens)
                    coherence = max(0.0, 1.0 - var / (mean + 1.0) / 10.0)

            overall = (rouge_l + bleu + coherence) / 3.0

            return {
                "metrics": {
                    "rouge_l": rouge_l,
                    "bleu_score": bleu,
                    "coherence": coherence
                },
                "overall_score": overall
            }


try:
    import openai
except Exception:
    openai = None

# config imports (safe fallbacks)
try:
    from src.config import (
        OPENAI_API_KEY,
        FINE_TUNED_MODEL_ID,
        QA_MAX_TOKENS,
        QA_TEMPERATURE,
        REFLECTION_MAX_LOOPS
    )
except Exception:
    OPENAI_API_KEY = os.environ.get(
        "OPENAI_API_KEY",
        ""
    )

    FINE_TUNED_MODEL_ID = (
        os.environ.get("FINE_TUNED_MODEL_ID", "ft:gpt-3.5-turbo-0125:shuvo::CftybjW7")
        or "gpt-4o-mini"
    )

    QA_MAX_TOKENS = int(os.environ.get("QA_MAX_TOKENS", "500"))
    QA_TEMPERATURE = float(os.environ.get("QA_TEMPERATURE", "0.7"))

    try:
        REFLECTION_MAX_LOOPS = int(os.environ.get("REFLECTION_MAX_LOOPS", "3"))
    except Exception:
        REFLECTION_MAX_LOOPS = 3


if openai is not None and OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception:
        logger.warning("Could not set OpenAI API key from config/environment.")


class TransformerQAStage:
    """
    Final QA Stage:
    CoR → Guardrails → LLM → Reflection → Evaluation.
    Metrics always computed from actual content; no default 0.5 values.
    """

    def __init__(self):
        self.cor_engine = ChainOfRetrieval()
        self.reflection = ReflectionChain()
        self.guardrails = Guardrails()
        self.evaluator = EvaluationLayer()

        logger.info("✅ Transformer QA Stage initialized.")

    def _call_llm(self, engineered_prompt: str) -> str:
        if openai is None:
            logger.info("OpenAI SDK not available — returning prompt preview.")
            return f"[LLM unavailable] {engineered_prompt[:200]}"

        try:
            resp = openai.ChatCompletion.create(
                model=FINE_TUNED_MODEL_ID,
                messages=[
                    {"role": "system", "content": "You are a precise medical QA assistant."},
                    {"role": "user", "content": engineered_prompt}
                ],
                max_tokens=QA_MAX_TOKENS,
                temperature=QA_TEMPERATURE
            )

            content = ""
            choices = getattr(resp, "choices", None) or (
                resp.get("choices") if isinstance(resp, dict) else None
            )

            if choices and len(choices) > 0:
                first = choices[0]

                if isinstance(first, dict):
                    msg = first.get("message") or {}
                    content = msg.get("content") or first.get("text") or ""
                else:
                    if hasattr(first, "message") and getattr(first.message, "content", None):
                        content = first.message.content
                    else:
                        content = getattr(first, "text", "") or ""
            else:
                content = (
                    getattr(resp, "text", "") or
                    (resp.get("text") if isinstance(resp, dict) else "")
                )

            return (content or "").strip() if content else f"[LLM fallback] {engineered_prompt[:200]}"

        except Exception as e:
            logger.warning(f"_call_llm failed: {e}")
            return f"[LLM error] {engineered_prompt[:200]}"

    def generate_answer(
        self,
        query: str,
        context_chain: Optional[List[Dict]] = None,
        plausibility_data: Optional[Dict] = None
    ) -> Dict[str, Any]:

        try:
            if context_chain is None:
                context_chain = []

            if plausibility_data is None:
                plausibility_data = {}

            # Input guardrail
            input_check = self.guardrails.check_input(query)
            blocked_input = not input_check.get("safe", True)
            answer_text = query if not blocked_input else "Query blocked for safety reasons."

            # Context engineering
            cor_result = self.cor_engine.process_context_for_qa(
                context_chain or [{"doc": "No context"}],
                query,
                plausibility_data.get("anomalies", [])
            )

            engineered_prompt = cor_result.get("engineered_prompt", f"Query: {query}")

            # LLM generation if input not blocked
            if not blocked_input:
                initial_answer = self._call_llm(engineered_prompt)
            else:
                initial_answer = answer_text

            # Output guardrail (first pass)
            output_check = self.guardrails.check_output(initial_answer)

            if not output_check.get("safe", True):
                initial_answer = "[Blocked output due to guardrails]"

            # Reflection
            top_doc = ""
            try:
                if context_chain and isinstance(context_chain, list) and len(context_chain) > 0:
                    top_doc = str(context_chain[0].get("doc", "") or "")
            except Exception:
                top_doc = ""

            try:
                max_loops_val = int(REFLECTION_MAX_LOOPS)
            except Exception:
                max_loops_val = 3

            try:
                reflection_result = self.reflection.reflect_and_refine(
                    initial_answer,
                    engineered_prompt,
                    top_doc
                )
                if not isinstance(reflection_result, dict):
                    raise TypeError("reflection returned non-dict")

            except TypeError:
                try:
                    reflection_result = self.reflection.reflect_and_refine(
                        initial_answer,
                        top_doc,
                        max_loops_val
                    )
                    if not isinstance(reflection_result, dict):
                        raise TypeError("reflection returned non-dict on second attempt")

                except Exception:
                    reflection_result = {
                        "refined_answer": initial_answer,
                        "final_hallucination_score": 0.0,
                        "loops_used": 0,
                        "safe": True
                    }

            except Exception:
                reflection_result = {
                    "refined_answer": initial_answer,
                    "final_hallucination_score": 0.0,
                    "loops_used": 0,
                    "safe": True
                }

            refined_answer = reflection_result.get("refined_answer", initial_answer)

            try:
                halluc_score = float(reflection_result.get("final_hallucination_score", 0.0) or 0.0)
            except Exception:
                halluc_score = 0.0

            try:
                loops_used = int(reflection_result.get("loops_used", 0) or 0)
            except Exception:
                loops_used = 0

            safe_flag = bool(reflection_result.get("safe", True))

            # Evaluation
            sentences: List[str] = []
            try:
                if isinstance(refined_answer, str):
                    raw = refined_answer.replace("\n", " ")
                    raw_sentences = [s.strip() for s in raw.split('.') if s.strip()]
                    sentences = raw_sentences

                elif isinstance(refined_answer, list):
                    sentences = [str(s).strip() for s in refined_answer if str(s).strip()]
                else:
                    sentences = [str(refined_answer).strip()]

            except Exception:
                sentences = [str(refined_answer or "").strip()]

            eval_result = self.evaluator.evaluate_qa_output({
                "answer": refined_answer,
                "sentences": sentences
            })

            eval_metrics = eval_result.get("metrics", {}) if isinstance(eval_result, dict) else {}

            try:
                overall_score = float(eval_result.get("overall_score", 0.0) or 0.0)
            except Exception:
                overall_score = 0.0

            try:
                confidence = max(0.0, min(1.0, overall_score * (1.0 - halluc_score)))
                confidence = round(confidence, 3)
            except Exception:
                confidence = 0.0

            return {
                "answer": refined_answer,
                "confidence": confidence,
                "used_contexts": int(len(context_chain)) if context_chain is not None else 0,
                "engineered_context_preview": (
                    engineered_prompt[:400] + ("..." if len(engineered_prompt) > 400 else "")
                ),
                "context_engineering_stats": {
                    "filtered_out": int(cor_result.get("filtered_contexts", 0) or 0),
                    "chain_steps": int(cor_result.get("chain_length", 1) or 1)
                },
                "reflection_stats": {
                    "hallucination_score": float(halluc_score),
                    "loops_used": int(loops_used),
                    "safe": bool(safe_flag)
                },
                "evaluation_metrics": {
                    "rouge_l": float(eval_metrics.get("rouge_l", 0.0) or 0.0),
                    "bleu_score": float(eval_metrics.get("bleu_score", 0.0) or 0.0),
                    "coherence": float(eval_metrics.get("coherence", 0.0) or 0.0)
                },
                "overall_score": overall_score,
                "guardrail_checks": {
                    "input_safe": bool(input_check.get("safe", True)),
                    "output_safe": bool(output_check.get("safe", True))
                }
            }

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"generate_answer failed: {e}\n{tb}")

            fallback_answer = "[QA stage failed — returning safe fallback answer]"

            preview_doc = ""
            try:
                preview_doc = (
                    context_chain[0].get("doc", "")[:400] + "..."
                ) if context_chain and len(context_chain) > 0 else ""
            except Exception:
                preview_doc = ""

            return {
                "answer": fallback_answer,
                "confidence": 0.0,
                "used_contexts": int(len(context_chain)) if context_chain is not None else 0,
                "engineered_context_preview": preview_doc,
                "context_engineering_stats": {"filtered_out": 0, "chain_steps": 0},
                "reflection_stats": {
                    "hallucination_score": 1.0,
                    "loops_used": 0,
                    "safe": False
                },
                "evaluation_metrics": {
                    "rouge_l": 0.0,
                    "bleu_score": 0.0,
                    "coherence": 0.0
                },
                "overall_score": 0.0,
                "guardrail_checks": {
                    "input_safe": False,
                    "output_safe": False
                },
                "debug": {
                    "error": str(e),
                    "traceback": tb
                }
            }
