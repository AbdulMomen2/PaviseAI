# Step 1: Clinical Normalizer (UMLS Integration via SciSpacy)
# =====================================================
import spacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
from typing import List, Dict
import re
from src.config import SCISPACY_MODEL

class ClinicalNormalizer:
    """
    Extracts and normalizes clinical entities from medical queries.
    Maps entities to UMLS Concept Unique Identifiers (CUIs).
    """
    def __init__(self, model_name: str = SCISPACY_MODEL):
        print(f"Loading SciSpacy model: {model_name}...")
        self.nlp = spacy.load(model_name)
        # Add entity linker for UMLS
        print("Adding UMLS entity linker...")
        self.nlp.add_pipe(
    "scispacy_linker",
    config={"resolve_abbreviations": True, "linker_name": "mesh"},
)

        
        # Add abbreviation detector
        self.nlp.add_pipe("abbreviation_detector")
        print("Clinical Normalizer initialized successfully!")

    def normalize_query(self, query: str) -> Dict:
        """
        Process a clinical query and extract normalized entities.
        Returns: Dictionary with original_query, normalized_query, entities, etc.
        """
        normalized_text = self._preprocess_text(query)
        doc = self.nlp(normalized_text)
        entities = self._extract_entities(doc)
        abbreviations = self._extract_abbreviations(doc)
        entity_summary = self._create_entity_summary(entities)
        return {
            "original_query": query,
            "normalized_query": normalized_text,
            "entities": entities,
            "abbreviations": abbreviations,
            "entity_summary": entity_summary,
            "is_medical_query": len(entities) > 0
        }

    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        abbreviation_map = {"pt": "patient", "dx": "diagnosis", "tx": "treatment", "hx": "history"}
        for abbr, expansion in abbreviation_map.items():
            text = re.sub(rf'\b{abbr}\b', expansion, text)
        return text

    def _extract_entities(self, doc) -> List[Dict]:
        entities = []
        for ent in doc.ents:
            entity_info = {
                "text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char,
                "cui": None, "cui_name": None, "confidence": 0.0, "semantic_types": []
            }
            if ent._.kb_ents:
                cui, confidence = ent._.kb_ents[0]
                entity_info["cui"] = cui
                entity_info["confidence"] = confidence
                linker = self.nlp.get_pipe("scispacy_linker")
                umls_entity = linker.kb.cui_to_entity.get(cui)
                if umls_entity:
                    entity_info["cui_name"] = umls_entity.canonical_name
                    entity_info["semantic_types"] = list(umls_entity.types)
            entities.append(entity_info)
        return entities

    def _extract_abbreviations(self, doc) -> List[Dict]:
        abbreviations = []
        if doc._.abbreviations:
            for abrv in doc._.abbreviations:
                abbreviations.append({
                    "abbreviation": abrv.text, "long_form": abrv._.long_form.text,
                    "start": abrv.start_char, "end": abrv.end_char
                })
        return abbreviations

    def _create_entity_summary(self, entities: List[Dict]) -> Dict:
        summary = {
            "total_entities": len(entities), "entity_types": {}, "semantic_categories": {},
            "has_disease": False, "has_symptom": False, "has_drug": False, "has_procedure": False
        }
        for entity in entities:
            label = entity["label"]
            summary["entity_types"][label] = summary["entity_types"].get(label, 0) + 1
            for sem_type in entity["semantic_types"]:
                summary["semantic_categories"][sem_type] = summary["semantic_categories"].get(sem_type, 0) + 1
                if sem_type in ["T047", "T048", "T191"]:
                    summary["has_disease"] = True
                elif sem_type in ["T184"]:
                    summary["has_symptom"] = True
                elif sem_type in ["T121", "T195", "T200"]:
                    summary["has_drug"] = True
                elif sem_type in ["T061"]:
                    summary["has_procedure"] = True
        return summary

    def get_cui_list(self, normalized_result: Dict) -> List[str]:
        return [entity["cui"] for entity in normalized_result["entities"] if entity["cui"] is not None]

    def format_for_downstream(self, normalized_result: Dict) -> str:
        query = normalized_result["normalized_query"]
        entities = normalized_result["entities"]
        annotated = query
        cui_mentions = [f"{entity['text']} [CUI:{entity['cui']}]" for entity in entities if entity["cui"]]
        if cui_mentions:
            annotated += " | Entities: " + ", ".join(cui_mentions)
        return annotated