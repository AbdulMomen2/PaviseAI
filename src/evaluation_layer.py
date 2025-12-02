# Evaluation Layer: Metrics for QA Output (Accuracy, ROUGE-L, F1, BERTScore, Coherence, BLEU Score)
# ===================================================================================================

import re
from typing import Dict, List, Optional, Any
import numpy as np
import logging

import spacy
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.config import EVALUATION_REFERENCE_ANSWERS


# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load spaCy English model (kept as your choice: en_core_sci_md)
# If model isn't installed, user must install it externally; this file assumes it's present.
nlp = spacy.load("en_core_sci_md")  # lightweight model for tokenization & sentence split


class EvaluationLayer:
    """
    Evaluation Layer using spaCy for tokenization and sentence splitting.
    Computes QA metrics:
    - Accuracy (Token Overlap)
    - ROUGE-L
    - F1 Score
    - BERTScore (semantic similarity via sentence-transformers)
    - Coherence (sentence similarity + lexical diversity)
    - BLEU Score
    """

    def __init__(self):
        # sentence-transformers model for semantic similarity
        self.bert_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.smoothing = SmoothingFunction().method4

        logger.info(
            "✅ Evaluation Layer initialized (spaCy tokenization, BLEU, ROUGE-L, BERTScore, Coherence)"
        )

    # ------------------------------
    # TEXT CLEANING / NORMALIZATION
    # ------------------------------
    @staticmethod
    def normalize_text(text: str) -> str:
        """Minimal normalization: lowercase, normalize whitespace, remove extra punctuation noise."""
        if text is None:
            return ""

        t = text.strip().lower()
        t = re.sub(r"[\r\n\t]+", " ", t)      # Replace newlines/tabs with space
        t = re.sub(r"\s+", " ", t)           # Normalize multiple spaces
        return t

    @staticmethod
    def strip_punctuation_tokens(tokens: List[str]) -> List[str]:
        """Remove purely punctuation tokens and trim punctuation around tokens."""
        cleaned = []
        for tok in tokens:
            tok = re.sub(r"^[^0-9a-zA-Z]+|[^0-9a-zA-Z]+$", "", tok)
            if tok:
                cleaned.append(tok)
        return cleaned

    # ------------------------------
    # TOKENIZATION
    # ------------------------------
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize using spaCy (lowercase, exclude space-like tokens)."""
        if not text:
            return []

        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_space]
        tokens = EvaluationLayer.strip_punctuation_tokens(tokens)
        return tokens

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split sentences using spaCy."""
        if not text:
            return []

        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # ------------------------------
    # METRICS
    # ------------------------------
    def compute_metrics(
        self,
        generated_answer: str,
        reference_answer: Optional[str] = None,
        sentences: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute all evaluation metrics for a single QA pair."""

        gen_text = self.normalize_text(generated_answer or "")
        ref_text = self.normalize_text(reference_answer or "")

        gen_tokens = self.tokenize(gen_text)
        ref_tokens = self.tokenize(ref_text)

        # ------------------------
        # Accuracy (Token Match vs generated tokens)
        # ------------------------
        if gen_tokens:
            if ref_tokens:
                correct = sum(1 for t in gen_tokens if t in ref_tokens)
                accuracy = correct / len(gen_tokens)
            else:
                accuracy = 0.0
        else:
            accuracy = 0.0

        # ------------------------
        # ROUGE-L
        # ------------------------
        try:
            if ref_text:
                rouge = self.scorer.score(ref_text, gen_text)
                rouge_l = rouge["rougeL"].fmeasure
            else:
                rouge_l = 0.0
        except Exception as e:
            logger.warning(f"ROUGE scoring failed: {e}")
            rouge_l = 0.0

        # ------------------------
        # F1 Score (Token Overlap)
        # ------------------------
        if gen_tokens and ref_tokens:
            set_gen = set(gen_tokens)
            set_ref = set(ref_tokens)

            overlap = len(set_gen & set_ref)
            precision = overlap / len(set_gen) if len(set_gen) > 0 else 0.0
            recall = overlap / len(set_ref) if len(set_ref) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
        else:
            f1 = 0.0

        # ------------------------
        # BERTScore (semantic similarity)
        # ------------------------
        try:
            gen_emb = self.bert_model.encode([gen_text], convert_to_tensor=True)
            if ref_text:
                ref_emb = self.bert_model.encode([ref_text], convert_to_tensor=True)
            else:
                ref_emb = None

            if ref_emb is not None:
                bert_score = util.cos_sim(gen_emb, ref_emb).item()
                bert_score = float(np.clip(bert_score, -1.0, 1.0))
                bert_score = (bert_score + 1.0) / 2.0
            else:
                bert_score = 0.0
        except Exception as e:
            logger.warning(f"BERT semantic scoring failed: {e}")
            bert_score = 0.0

        # ------------------------
        # Coherence Score
        # ------------------------
        coherence = 0.0

        if not sentences:
            sentences = self.split_sentences(gen_text)

        if sentences and len(sentences) > 1:
            try:
                sent_embs = self.bert_model.encode(sentences, convert_to_tensor=True)
                sim_sum = 0.0
                pairs = 0

                for i in range(1, len(sent_embs)):
                    sim = util.cos_sim(
                        sent_embs[i - 1 : i], sent_embs[i : i + 1]
                    ).item()
                    sim_sum += sim
                    pairs += 1

                avg_sent_similarity = sim_sum / pairs if pairs > 0 else 0.0
                lexical_diversity = (
                    len(set(gen_tokens)) / len(gen_tokens) if gen_tokens else 0.0
                )

                avg_sent_similarity = (avg_sent_similarity + 1.0) / 2.0
                coherence = float(
                    (avg_sent_similarity + lexical_diversity) / 2.0
                )
            except Exception as e:
                logger.warning(f"Coherence computation failed: {e}")
                coherence = 0.0

        # ------------------------
        # BLEU Score
        # ------------------------
        bleu_score = 0.0
        try:
            if ref_tokens and gen_tokens:
                bleu_score = sentence_bleu(
                    [ref_tokens], gen_tokens, smoothing_function=self.smoothing
                )
            else:
                bleu_score = 0.0
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            bleu_score = 0.0

        return {
            "accuracy": float(np.clip(accuracy, 0.0, 1.0)),
            "rouge_l": float(np.clip(rouge_l, 0.0, 1.0)),
            "f1_score": float(np.clip(f1, 0.0, 1.0)),
            "bert_score": float(np.clip(bert_score, 0.0, 1.0)),
            "coherence": float(np.clip(coherence, 0.0, 1.0)),
            "bleu_score": float(np.clip(bleu_score, 0.0, 1.0)),
        }

    # ------------------------------
    # OVERALL EVALUATION
    # ------------------------------
    def evaluate_qa_output(self, qa_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full QA Evaluation returning all metrics and overall score.

        qa_result may contain:
          - 'answer'
          - 'reference_answer'
          - 'id' / 'question'
        """

        answer_text = qa_result.get("answer", "") or ""

        reference_answer = qa_result.get("reference_answer")

        if not reference_answer:
            fallback_keys = [
                qa_result.get("id"),
                qa_result.get("question"),
                qa_result.get("prompt"),
            ]
            for key in fallback_keys:
                if key and key in EVALUATION_REFERENCE_ANSWERS:
                    reference_answer = EVALUATION_REFERENCE_ANSWERS.get(key)
                    break

        if not reference_answer:
            prefix = answer_text[:60]
            reference_answer = EVALUATION_REFERENCE_ANSWERS.get(prefix, "")

        sentences = self.split_sentences(answer_text)

        metrics = self.compute_metrics(
            answer_text, reference_answer=reference_answer, sentences=sentences
        )

        overall_score = float(
            np.mean(
                [
                    metrics["accuracy"],
                    metrics["rouge_l"],
                    metrics["f1_score"],
                    metrics["bert_score"],
                    metrics["coherence"],
                    metrics["bleu_score"],
                ]
            )
        )

        logger.info(
            f"📊 QA Evaluation: Overall {overall_score:.3f}, "
            f"ROUGE-L {metrics['rouge_l']:.3f}, BLEU {metrics['bleu_score']:.3f}, "
            f"Coherence {metrics['coherence']:.3f}"
        )

        return {
            "metrics": metrics,
            "overall_score": overall_score,
        }
