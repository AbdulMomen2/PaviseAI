# Evaluation Layer: Metrics for QA Output (Accuracy, ROUGE-L, F1, BERTScore, Confusion Matrix, Coherence, BLEU Score)
# =====================================================
import spacy
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from src.config import EVALUATION_REFERENCE_ANSWERS

# Load spaCy English model
nlp = spacy.load("en_core_sci_md")  # lightweight model for tokenization & sentence split

class EvaluationLayer:
    """
    Evaluation Layer using spaCy for tokenization and sentence splitting.
    Computes QA metrics:
    - Accuracy: Token-level exact match
    - ROUGE-L
    - F1 Score
    - BERTScore
    - Confusion Matrix
    - Coherence (Sentence similarity + Lexical diversity)
    - BLEU Score
    """
    def __init__(self):
        self.bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method4
        print("✅ Evaluation Layer initialized (spaCy tokenization, BLEU, ROUGE-L, BERTScore, Coherence)")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize using spaCy (lowercase, exclude spaces)."""
        doc = nlp(text)
        return [token.text.lower() for token in doc if not token.is_space]

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split sentences using spaCy."""
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def compute_metrics(self, generated_answer: str, reference_answer: str = None, sentences: List[str] = None) -> Dict:
        """Compute all evaluation metrics for a single QA pair."""
        if not reference_answer:
            reference_answer = EVALUATION_REFERENCE_ANSWERS.get(generated_answer[:50], "")

        # Tokenize
        gen_tokens = self.tokenize(generated_answer)
        ref_tokens = [self.tokenize(reference_answer)] if reference_answer else [[]]

        # Accuracy (exact match)
        if ref_tokens[0]:
            accuracy = accuracy_score(
                [1 if t in ref_tokens[0] else 0 for t in gen_tokens],
                [1] * len(gen_tokens)
            )
        else:
            accuracy = 0.0

        # ROUGE-L
        rouge = self.scorer.score(reference_answer, generated_answer)
        rouge_l = rouge['rougeL'].fmeasure

        # F1 Score (token overlap)
        y_true = [1 if t in ref_tokens[0] else 0 for t in gen_tokens]
        y_pred = [1] * len(gen_tokens)
        f1 = f1_score(y_true, y_pred, average='binary') if y_true else 0.0

        # BERTScore (semantic similarity)
        gen_emb = self.bert_model.encode([generated_answer])
        ref_emb = self.bert_model.encode([reference_answer]) if reference_answer else gen_emb
        bert_score = util.cos_sim(gen_emb, ref_emb).item()

        # Confusion Matrix
        try:
            cm = confusion_matrix(y_true, y_pred).tolist()
        except Exception:
            cm = [[0,0],[0,0]]

        # Enhanced Coherence: sentence similarity + lexical diversity
        coherence = 0.0
        lexical_diversity = 0.0
        if sentences and len(sentences) > 1:
            sent_embs = self.bert_model.encode(sentences)
            sim_sum = 0.0
            for i in range(1, len(sent_embs)):
                sim_sum += util.cos_sim(sent_embs[i-1:i], sent_embs[i:i+1]).item()
            coherence = sim_sum / (len(sentences) - 1)
            lexical_diversity = len(set(gen_tokens)) / len(gen_tokens) if gen_tokens else 0
        coherence = (coherence + lexical_diversity) / 2

        # BLEU Score
        try:
            bleu_score = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=self.smoothing)
        except Exception:
            bleu_score = 0.0

        return {
            "accuracy": accuracy,
            "rouge_l": rouge_l,
            "f1_score": f1,
            "bert_score": bert_score,
            "confusion_matrix": cm,
            "coherence": coherence,
            "bleu_score": bleu_score
        }

    def evaluate_qa_output(self, qa_result: Dict) -> Dict:
        """Full QA Evaluation returning all metrics and overall score."""
        answer_text = qa_result.get('answer', "")
        sentences = self.split_sentences(answer_text)
        metrics = self.compute_metrics(answer_text, sentences=sentences)
        overall_score = np.mean([
            metrics['accuracy'],
            metrics['rouge_l'],
            metrics['f1_score'],
            metrics['bert_score'],
            metrics['coherence'],
            metrics['bleu_score']
        ])
        print(f"📊 QA Evaluation: Overall {overall_score:.3f}, ROUGE-L {metrics['rouge_l']:.3f}, BLEU {metrics['bleu_score']:.3f}, Coherence {metrics['coherence']:.3f}")
        return {"metrics": metrics, "overall_score": overall_score}
