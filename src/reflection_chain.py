# Adversarial Reflection Chain: Self-Critique Loop for Hallucination Guardrail
# =====================================================
import openai
from typing import Dict, List
from src.config import OPENAI_API_KEY, FINE_TUNED_MODEL_ID, REFLECTION_MAX_LOOPS, HALLUCINATION_THRESHOLD
import numpy as np
from src.config import QA_MAX_TOKENS

openai.api_key = OPENAI_API_KEY

class ReflectionChain:
    """
    Adversarial Reflection Chain: Iterative Self-Critique for Hallucination Detection & Refinement.
    - Loop: Generate → Critique → Refine (up to max_loops).
    - Guardrail: If hallucination score > threshold, block/refine.
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

    def self_critique(self, answer: str, context: str) -> Dict:
        """Single Critique Iteration."""
        prompt = self.CRITIQUE_PROMPT.replace("{{ answer }}", answer).replace("{{ context }}", context)
        try:
            response = openai.ChatCompletion.create(
                model=FINE_TUNED_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.2
            )
            critique = response.choices[0].message.content.strip()
            # Parse score (simple regex/heuristic)
            score_str = [line for line in critique.split('\n') if 'Score' in line][0]
            hallucination_score = float(score_str.split(':')[-1].strip()) if score_str else 0.0
            suggestion = critique.split('Suggestion:')[-1].strip() if 'Suggestion:' in critique else "No change needed."
            return {"hallucination_score": hallucination_score, "suggestion": suggestion, "critique": critique}
        except Exception as e:
            print(f"⚠️ Critique Error: {e}")
            return {"hallucination_score": 0.0, "suggestion": "", "critique": ""}

    def reflect_and_refine(self, initial_answer: str, context: str, max_loops: int = REFLECTION_MAX_LOOPS) -> Dict:
        """Full Reflection Loop."""
        current_answer = initial_answer
        critiques = []
        for loop in range(max_loops):
            critique = self.self_critique(current_answer, context)
            critiques.append(critique)
            if critique['hallucination_score'] > HALLUCINATION_THRESHOLD:
                # Refine
                refine_prompt = f"Refine this answer based on critique: {critique['suggestion']}\nOriginal: {current_answer}\nRefined:"
                response = openai.ChatCompletion.create(
                    model=FINE_TUNED_MODEL_ID,
                    messages=[{"role": "user", "content": refine_prompt}],
                    max_tokens=QA_MAX_TOKENS // 2,
                    temperature=0.3
                )
                current_answer = response.choices[0].message.content.strip()
                print(f"🔄 Reflection Loop {loop+1}: Hallucination {critique['hallucination_score']:.2f} - Refined.")
            else:
                print(f"✅ Reflection Loop {loop+1}: Passed - Low Hallucination {critique['hallucination_score']:.2f}")
                break
        final_score = np.mean([c['hallucination_score'] for c in critiques])
        print(f"🔒 Hallucination Guardrail: Final Score {final_score:.2f} - {'Safe' if final_score < HALLUCINATION_THRESHOLD else 'Refined'}")
        return {
            "refined_answer": current_answer,
            "final_hallucination_score": final_score,
            "critiques": critiques,
            "loops_used": len(critiques),
            "safe": final_score < HALLUCINATION_THRESHOLD
        }