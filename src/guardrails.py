# Guardrails: Input/Output Rails with Self-Check Prompts (NeMo-like)
# =====================================================
import openai
from typing import Dict

from src.config import OPENAI_API_KEY, FINE_TUNED_MODEL_ID, GUARDRAILS_INPUT_PROMPT, GUARDRAILS_OUTPUT_PROMPT

openai.api_key = OPENAI_API_KEY

class Guardrails:
    """
    Guardrails Module: Input/Output Checks for Safety (Harmful, Abusive, Explicit Content).
    Uses LLM-based self-check prompts (NeMo Guardrails style).
    """

    def __init__(self):
        print("✅ Guardrails initialized (Input/Output Rails)")

    def check_input(self, user_input: str) -> Dict:
        """Check User Input for Violations."""
        prompt = GUARDRAILS_INPUT_PROMPT.replace("{{ user_input }}", user_input)
        try:
            response = openai.ChatCompletion.create(
                model=FINE_TUNED_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )

            choice = response.choices[0]
            content = getattr(choice.message, "content", None) if hasattr(choice, "message") else choice.get("text", "")
            check_result = content.strip() if content else ""

            # Block detection
            blocked = "Yes" in check_result or "Blocked" in check_result

            # Extract reason
            if "Reason:" in check_result:
                reason = check_result.split("Reason:")[-1].strip()
            else:
                reason = "Input violates policy." if blocked else "Passed guardrail check."

            print(f"🔒 Input Guardrail: {'Blocked' if blocked else 'Passed'} - {reason[:50]}...")
            return {"blocked": blocked, "reason": reason, "safe": not blocked}

        except Exception as e:
            print(f"⚠️ Input Check Error: {e}")
            return {"blocked": True, "reason": f"Check failed due to error: {e}", "safe": False}

    def check_output(self, bot_response: str) -> Dict:
        """Check Bot Output for Violations."""
        prompt = GUARDRAILS_OUTPUT_PROMPT.replace("{{ bot_response }}", bot_response)
        try:
            response = openai.ChatCompletion.create(
                model=FINE_TUNED_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )

            choice = response.choices[0]
            content = getattr(choice.message, "content", None) if hasattr(choice, "message") else choice.get("text", "")
            check_result = content.strip() if content else ""

            blocked = "Yes" in check_result or "Blocked" in check_result

            if "Reason:" in check_result:
                reason = check_result.split("Reason:")[-1].strip()
            else:
                reason = "Output violates policy." if blocked else "Passed guardrail check."

            print(f"🔒 Output Guardrail: {'Blocked' if blocked else 'Passed'} - {reason[:50]}...")
            return {"blocked": blocked, "reason": reason, "safe": not blocked}

        except Exception as e:
            print(f"⚠️ Output Check Error: {e}")
            return {"blocked": True, "reason": f"Check failed due to error: {e}", "safe": False}
