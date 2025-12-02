# =====================================================
# Step 2: Prompt Injection Detector (DistilBERT Version)
# =====================================================

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from dataclasses import dataclass
import re
from typing import Dict
from src.config import DEBERTA_MODEL, DEVICE, PATTERN_THRESHOLD, MODEL_THRESHOLD, COMBINED_THRESHOLD

# ------------------ Injection Pattern Class ------------------
@dataclass
class InjectionPattern:
    JAILBREAK_PATTERNS = [
        r"ignore previous instructions",
        r"disregard (all|any|the) (previous|above|prior) (instructions|prompts|rules)",
        r"you are now (a|an|in) (different|new|alternate|DAN|unrestricted)",
        r"roleplay as",
        r"pretend (you are|to be|you're)",
        r"act as (if|though)?",
        r"forget (everything|all|your) (previous|prior|earlier)",
        r"new (instructions|guidelines|rules|system)",
        r"override (your|the) (safety|ethical|previous)",
        r"bypass (your|the|all) (restrictions|guidelines|safety)",
        r"system prompt",
        r"developer mode",
        r"god mode",
        r"admin mode",
        r"sudo mode"
    ]

    PROMPT_LEAK_PATTERNS = [
        r"(show|reveal|display|tell|give|print) (me )?(your|the) (system |original )?(prompt|instructions)",
        r"what (are|were) (your|the) (initial|original|system) (instructions|prompt)",
        r"repeat (your|the) (instructions|prompt|rules)",
        r"(system|initial|original) message",
        r"what did (i|they|the user) tell you",
        r"your guidelines",
        r"configuration settings"
    ]

    MALICIOUS_MEDICAL_PATTERNS = [
        r"how to (make|create|synthesize|manufacture) (illegal |illicit )?(drugs|narcotics|controlled substances)",
        r"(create|make|synthesize) (poison|toxin|harmful substance)",
        r"(self|commit|attempt) (harm|suicide|murder)",
        r"overdose",
        r"euthanasia (methods|instructions|how)",
        r"(fake|forge|counterfeit) (prescription|medical records)",
        r"undetectable (poison|drug)",
        r"lethal dose"
    ]

    INJECTION_INDICATORS = [
        r"\[INST\]",
        r"\[/INST\]",
        r"<\|.*?\|>",
        r"{{.*?}}",
        r"###\s*(System|User|Assistant)",
        r"---\s*New (conversation|chat|session)",
        r"</?(system|prompt|instructions)>",
        r"BEGIN NEW (SESSION|CONVERSATION|INSTRUCTIONS)"
    ]

print("✅ Injection patterns defined!")

# ------------------ MAIN DETECTOR CLASS ------------------
class PromptInjectionDetector:
    def __init__(self, model_name: str = "distilbert-base-uncased", use_lora: bool = True, device: str = DEVICE):
        print(f"🔄 Initializing Prompt Injection Detector...")
        self.device = device

        # DistilBERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # DistilBERT classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )

        # Apply LoRA if enabled
        if use_lora:
            self._apply_lora()

        self.model.to(device)
        self.model.eval()

        # config values
        self.patterns = InjectionPattern()
        self.pattern_threshold = PATTERN_THRESHOLD
        self.model_threshold = MODEL_THRESHOLD
        self.combined_threshold = COMBINED_THRESHOLD

        print("✅ Prompt Injection Detector initialized successfully!")

    # ------------------ LoRA ------------------
    def _apply_lora(self):
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)
        print(f"✅ LoRA applied - Trainable parameters: {self.model.print_trainable_parameters()}")

    # ------------------ PATTERN DETECTOR ------------------
    def _pattern_based_detection(self, query: str) -> Dict:
        query_lower = query.lower()
        flagged_patterns = []
        pattern_scores = []

        for pattern in self.patterns.JAILBREAK_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                flagged_patterns.append(("jailbreak", pattern))
                pattern_scores.append(1.0)

        for pattern in self.patterns.PROMPT_LEAK_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                flagged_patterns.append(("prompt_leak", pattern))
                pattern_scores.append(0.9)

        for pattern in self.patterns.MALICIOUS_MEDICAL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                flagged_patterns.append(("malicious_medical", pattern))
                pattern_scores.append(0.95)

        for pattern in self.patterns.INJECTION_INDICATORS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                flagged_patterns.append(("injection_indicator", pattern))
                pattern_scores.append(0.85)

        pattern_score = max(pattern_scores) if pattern_scores else 0.0

        return {
            "method": "pattern_based",
            "risk_score": pattern_score,
            "is_safe": pattern_score < self.pattern_threshold,
            "flagged_patterns": flagged_patterns,
            "confidence": 0.9 if flagged_patterns else 0.5
        }

    # ------------------ MODEL DETECTOR ------------------
    def _model_based_detection(self, query: str) -> Dict:
        try:
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                safe_prob = probabilities[0][0].item()
                unsafe_prob = probabilities[0][1].item()
        except Exception:
            # fallback in case of tokenization/model errors
            safe_prob, unsafe_prob = 0.5, 0.5

        return {
            "method": "model_based",
            "risk_score": unsafe_prob,
            "is_safe": safe_prob > unsafe_prob,
            "confidence": max(safe_prob, unsafe_prob),
            "safe_probability": safe_prob,
            "unsafe_probability": unsafe_prob
        }

    # ------------------ CONTEXT DETECTOR ------------------
    def _context_aware_detection(self, query: str, normalized_result: Dict = None) -> Dict:
        if normalized_result is None:
            return {
                "method": "context_aware",
                "risk_score": 0.0,
                "is_safe": True,
                "confidence": 0.3,
                "note": "No normalization context provided"
            }

        risk_signals = []
        risk_score = 0.0
        has_medical_entities = normalized_result.get("is_medical_query", False)
        entity_count = len(normalized_result.get("entities", []))
        words = query.split()

        if len(words) > 20 and entity_count < 2:
            risk_signals.append("Low medical entity density in long query")
            risk_score += 0.3

        context_switch_patterns = [
            r"(but|however|instead|actually|now)\s+(ignore|forget|disregard)",
            r"medical.*?(ignore|forget|override)",
            r"(diabetes|cancer|heart).*?(system prompt|instructions)"
        ]

        for pattern in context_switch_patterns:
            if re.search(pattern, query.lower()):
                risk_signals.append(f"Context switching detected: {pattern}")
                risk_score += 0.4

        special_char_ratio = sum(1 for c in query if not c.isalnum() and c != ' ') / max(len(query), 1)

        if special_char_ratio > 0.15:
            risk_signals.append(f"High special character ratio: {special_char_ratio:.2f}")
            risk_score += 0.2

        risk_score = min(risk_score, 1.0)

        return {
            "method": "context_aware",
            "risk_score": risk_score,
            "is_safe": risk_score < 0.5,
            "confidence": 0.7,
            "risk_signals": risk_signals,
            "has_medical_context": has_medical_entities,
            "entity_count": entity_count
        }

    # ------------------ COMBINE DETECTION ------------------
    def _combine_detections(self, query: str, pattern_result: Dict, model_result: Dict, context_result: Dict) -> Dict:

        weights = {"pattern": 0.4, "model": 0.4, "context": 0.2}

        combined_risk_score = (
            pattern_result["risk_score"] * weights["pattern"] +
            model_result["risk_score"] * weights["model"] +
            context_result["risk_score"] * weights["context"]
        )

        combined_confidence = (
            pattern_result["confidence"] * weights["pattern"] +
            model_result["confidence"] * weights["model"] +
            context_result["confidence"] * weights["context"]
        )

        is_safe = combined_risk_score < self.combined_threshold

        if pattern_result["risk_score"] >= self.pattern_threshold:
            primary_method = "pattern_based"
            reason = f"Matched injection patterns: {[p[1] for p in pattern_result['flagged_patterns'][:3]]}"

        elif model_result["risk_score"] >= self.model_threshold:
            primary_method = "model_based"
            reason = f"Model detected unsafe content (confidence: {model_result['confidence']:.2f})"

        elif context_result["risk_score"] >= 0.5:
            primary_method = "context_aware"
            reason = f"Context analysis flagged risks: {context_result.get('risk_signals', [])[:2]}"

        else:
            primary_method = "combined"
            reason = "All checks passed - query appears safe"

        if combined_risk_score >= 0.9:
            recommendation = "BLOCK"
            action = "Reject query immediately - high confidence attack"
        elif combined_risk_score >= 0.75:
            recommendation = "REJECT"
            action = "Reject query - likely prompt injection"
        elif combined_risk_score >= 0.5:
            recommendation = "WARN"
            action = "Flag for review - suspicious patterns detected"
        else:
            recommendation = "ALLOW"
            action = "Proceed to next step - query appears safe"

        return {
            "query": query,
            "is_safe": is_safe,
            "risk_score": combined_risk_score,
            "confidence": combined_confidence,
            "primary_detection_method": primary_method,
            "reason": reason,
            "recommendation": recommendation,
            "action": action,
            "detailed_results": {
                "pattern_based": pattern_result,
                "model_based": model_result,
                "context_aware": context_result
            }
        }

    # ------------------ PUBLIC METHOD ------------------
    def detect_injection(self, query: str, normalized_result: Dict = None) -> Dict:
        pattern_result = self._pattern_based_detection(query)
        model_result = self._model_based_detection(query)
        context_result = self._context_aware_detection(query, normalized_result)
        final_result = self._combine_detections(query, pattern_result, model_result, context_result)
        return final_result
