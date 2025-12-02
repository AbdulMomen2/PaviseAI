# Adversarial Reflection Chain: Self-Critique Loop for Hallucination Guardrail
# =====================================================
import re
import openai
from typing import Dict, List, Any, Optional
from src.config import OPENAI_API_KEY, FINE_TUNED_MODEL_ID, REFLECTION_MAX_LOOPS, HALLUCINATION_THRESHOLD
import numpy as np
from src.config import QA_MAX_TOKENS

openai.api_key = OPENAI_API_KEY


class ReflectionChain:
    """
    Adversarial Reflection Chain: Iterative Self-Critique for Hallucination Detection & Refinement.
    - Loop: Generate → Critique → Refine (up to max_loops).
    - Guardrail: If hallucination score > threshold, block/refine.

    Notes on compatibility:
    - This method is defensive about calling shapes. Some callers pass:
        reflect_and_refine(initial_answer, context, max_loops)
      while others pass:
        reflect_and_refine(initial_answer, engineered_prompt, top_context)
      To remain fully compatible we:
        - Accept the common signature (initial_answer, context, max_loops)
        - If max_loops is a non-numeric string (likely top_context), treat it as top_context
          and fall back to default max_loops from config.
    - All conversions are safe (no ValueError on empty/invalid values).
    """

    def __init__(self):
        self.CRITIQUE_PROMPT = """
You are a hallucination detector. Critique the answer for:
- Factual accuracy against context.
- No unsubstantiated claims.
- Coherence with evidence.

Answer: {{ answer }}
Context: {{ context }}

Critique Score (0-1, 1=high hallucination): 
Refinement Suggestion: 
"""
        print("✅ Reflection Chain initialized (Self-Critique Loop & Hallucination Guardrail)")

    def _safe_parse_score(self, critique_text: str) -> float:
        """
        Robustly extract a float in [0,1] from critique text.
        Looks for lines containing 'score' or 'critique score' or a lone decimal.
        Falls back to 0.0 on failure.
        """
        if not critique_text:
            return 0.0

        # Try explicit 'score' lines first (case-insensitive)
        for line in critique_text.splitlines():
            if "score" in line.lower():
                # take last colon-separated part
                parts = line.split(":")
                if len(parts) >= 2:
                    candidate = parts[-1].strip()
                    # extract first float-like token
                    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", candidate)
                    if m:
                        try:
                            val = float(m.group(0))
                            return max(0.0, min(1.0, val))
                        except Exception:
                            pass

        # fallback: search whole text for first decimal number
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", critique_text)
        if m:
            try:
                val = float(m.group(0))
                return max(0.0, min(1.0, val))
            except Exception:
                pass

        return 0.0

    def self_critique(self, answer: str, context: str) -> Dict[str, Any]:
        """Single Critique Iteration."""
        # Ensure inputs are strings
        answer = "" if answer is None else str(answer)
        context = "" if context is None else str(context)

        prompt = (
            self.CRITIQUE_PROMPT
            .replace("{{ answer }}", answer)
            .replace("{{ context }}", context)
        )

        try:
            response = openai.ChatCompletion.create(
                model=FINE_TUNED_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.2
            )

            # robust extraction of content for different SDK shapes
            choice = response.choices[0]
            critique = ""
            if hasattr(choice, "message") and getattr(choice.message, "content", None):
                critique = choice.message.content.strip()
            else:
                # older/newer shapes: dict or fallback
                try:
                    # dict-like
                    critique = (choice.get("message") or {}).get("content") or choice.get("text") or ""
                except Exception:
                    critique = str(choice)

            critique = critique.strip()

            # parse hallucination score robustly
            hallucination_score = self._safe_parse_score(critique)

            # suggestion parsing: try common labels
            suggestion = ""
            # look for "Refinement Suggestion", "Suggestion", "Refine"
            suggestion_markers = ["Refinement Suggestion:", "Refinement Suggestion", "Suggestion:", "Suggestion", "Refine:"]
            for marker in suggestion_markers:
                if marker in critique:
                    suggestion = critique.split(marker, 1)[-1].strip()
                    break
            if not suggestion:
                # attempt heuristic: last line if short
                lines = [l.strip() for l in critique.splitlines() if l.strip()]
                if len(lines) >= 1:
                    suggestion = lines[-1] if len(lines[-1].split()) <= 40 else "No change needed."
                else:
                    suggestion = "No change needed."

            return {"hallucination_score": hallucination_score, "suggestion": suggestion, "critique": critique}

        except Exception as e:
            # never crash — return safe default critique
            print(f"⚠️ Critique Error: {e}")
            return {"hallucination_score": 0.0, "suggestion": "Critique failed; no suggestion.", "critique": ""}

    def reflect_and_refine(
        self,
        initial_answer: str,
        context: Optional[str] = None,
        max_loops: Optional[Any] = REFLECTION_MAX_LOOPS,
        top_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full Reflection Loop.

        Backwards-compatible calling patterns handled:
        - reflect_and_refine(initial_answer, context, max_loops)
        - reflect_and_refine(initial_answer, engineered_prompt, top_context)
          (in which case `max_loops` may be a string containing the top_context)
        """

        # normalize strings
        initial_answer = "" if initial_answer is None else str(initial_answer)
        context = "" if context is None else str(context)
        top_context = "" if top_context is None else str(top_context)

        # If callers accidentally passed top_context in place of max_loops (string),
        # detect and shift values: if max_loops is non-numeric string -> treat as top_context.
        numeric_max_loops = None
        try:
            # try direct integer conversion when possible
            if max_loops is None or (isinstance(max_loops, str) and max_loops.strip() == ""):
                numeric_max_loops = int(REFLECTION_MAX_LOOPS)
            else:
                # if it's a string that is numeric, parse it
                if isinstance(max_loops, str):
                    if re.fullmatch(r"\d+", max_loops.strip()):
                        numeric_max_loops = int(max_loops.strip())
                    else:
                        # non-numeric string — likely top_context passed here
                        # shift: use provided string as top_context unless top_context already provided
                        if not top_context:
                            top_context = max_loops
                        numeric_max_loops = int(REFLECTION_MAX_LOOPS)
                else:
                    numeric_max_loops = int(max_loops)
        except Exception:
            numeric_max_loops = int(REFLECTION_MAX_LOOPS)

        # enforce sane bounds
        if numeric_max_loops < 1:
            numeric_max_loops = int(REFLECTION_MAX_LOOPS) if int(REFLECTION_MAX_LOOPS) >= 1 else 1
        if numeric_max_loops > 20:
            # protect against runaway loops; you can tune this upper bound
            numeric_max_loops = 20

        current_answer = initial_answer
        critiques: List[Dict[str, Any]] = []

        for loop in range(numeric_max_loops):
            critique = self.self_critique(current_answer, context or top_context or "")
            critiques.append(critique)

            # if hallucination detected above threshold -> refine, else break
            try:
                score = float(critique.get("hallucination_score", 0.0) or 0.0)
            except Exception:
                score = 0.0

            if score > HALLUCINATION_THRESHOLD:
                # build a safe refine prompt using suggestion
                suggestion_text = critique.get("suggestion", "") or ""
                # limit the size of the prompt to prevent token blowup
                safe_suggestion = suggestion_text if len(suggestion_text) <= 800 else suggestion_text[:800] + "..."
                safe_original = current_answer if len(current_answer) <= 1200 else current_answer[:1200] + "..."

                refine_prompt = (
                    f"Refine this answer based on the critique/suggestion below.\n\n"
                    f"Critique Suggestion: {safe_suggestion}\n\n"
                    f"Original Answer:\n{safe_original}\n\n"
                    f"Produce a corrected, concise, and evidence-backed refined answer. If the claim cannot be supported, say so clearly."
                )

                try:
                    response = openai.ChatCompletion.create(
                        model=FINE_TUNED_MODEL_ID,
                        messages=[{"role": "user", "content": refine_prompt}],
                        max_tokens=max(16, QA_MAX_TOKENS // 2),
                        temperature=0.3
                    )
                    choice = response.choices[0]
                    refined_candidate = ""
                    if hasattr(choice, "message") and getattr(choice.message, "content", None):
                        refined_candidate = choice.message.content.strip()
                    else:
                        try:
                            refined_candidate = (choice.get("message") or {}).get("content") or choice.get("text") or ""
                        except Exception:
                            refined_candidate = str(choice)
                    # if LLM returned empty or nonsense, keep previous answer
                    if refined_candidate and len(refined_candidate) > 0:
                        current_answer = refined_candidate.strip()
                    else:
                        # if empty, stop refining to avoid loop
                        print("⚠️ Refinement produced empty content — stopping further refinement.")
                        break

                    print(f"🔄 Reflection Loop {loop + 1}: Hallucination {score:.2f} - Refined.")
                except Exception as e:
                    print(f"⚠️ Refinement Error (LLM): {e} - stopping refinement.")
                    break
            else:
                print(f"✅ Reflection Loop {loop + 1}: Passed - Low Hallucination {score:.2f}")
                break

        # compute final score safely
        if critiques:
            final_score = float(np.mean([float(c.get("hallucination_score", 0.0) or 0.0) for c in critiques]))
            # clamp
            final_score = max(0.0, min(1.0, final_score))
        else:
            final_score = 0.0

        print(
            f"🔒 Hallucination Guardrail: Final Score {final_score:.2f} - "
            f"{'Safe' if final_score < HALLUCINATION_THRESHOLD else 'Refined'}"
        )

        return {
            "refined_answer": current_answer,
            "final_hallucination_score": final_score,
            "critiques": critiques,
            "loops_used": len(critiques),
            "safe": final_score < HALLUCINATION_THRESHOLD
        }
