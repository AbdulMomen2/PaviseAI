# Data Store: Neo4j Handler (UMLS Graph + Real PubMed + PDF/URL Integration)
# =====================================================
from neo4j import GraphDatabase, Session
from Bio import Entrez
from typing import List, Dict, Optional, Any
import time
import logging
import pdfplumber
import requests
from bs4 import BeautifulSoup
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NCBI_EMAIL, NEO4J_DATABASE

# configure simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Entrez.email = NCBI_EMAIL

class Neo4jHandler:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        database: str = None,
    ):
        self.database = database or (NEO4J_DATABASE if "NEO4J_DATABASE" in globals() else "neo4j")
        logger.info("🔄 Connecting to Neo4j...")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            def _test(tx):
                return tx.run("RETURN 1 AS ok").single().get("ok")
            val = self.driver.session(database=self.database).execute_read(_test)
            logger.info("✅ Neo4j connected successfully! (db=%s) result=%s", self.database, val)
        except Exception as e:
            logger.exception("❌ Failed to connect to Neo4j: %s", e)
            raise

    def close(self):
        try:
            self.driver.close()
            logger.info("Closed Neo4j driver.")
        except Exception:
            pass

    # -------------------------
    # Utility helpers
    # -------------------------
    @staticmethod
    def _safe_query_id(text: str, max_len: int = 50) -> str:
        if not text:
            return "empty_query"
        q = text.strip().replace("\n", " ")
        return q[:max_len]

    @staticmethod
    def _extract_text_from_pdf(pdf_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.warning("Failed to extract text from PDF %s: %s", pdf_path, e)
        return text.strip()

    @staticmethod
    def _extract_text_from_url(url: str) -> str:
        text = ""
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for s in soup(["script", "style"]):
                    s.decompose()
                text = soup.get_text(separator="\n")
        except Exception as e:
            logger.warning("Failed to extract text from URL %s: %s", url, e)
        return text.strip()

    # -------------------------
    # PDF / URL storage
    # -------------------------
    def store_documents(self, pdf_paths: List[str] = None, urls: List[str] = None) -> None:
        pdf_paths = pdf_paths or []
        urls = urls or []

        def _tx(tx, doc_id: str, source: str, content: str):
            tx.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.source = $source,
                    d.content = $content,
                    d.updated = datetime()
                """,
                doc_id=doc_id,
                source=source,
                content=content
            )

        for pdf_path in pdf_paths:
            doc_id = self._safe_query_id(pdf_path)
            content = self._extract_text_from_pdf(pdf_path)
            if not content:
                logger.warning("PDF %s has no extractable text, skipping.", pdf_path)
                continue
            try:
                with self.driver.session(database=self.database) as session:
                    session.execute_write(_tx, doc_id, f"PDF:{pdf_path}", content)
                logger.info("💾 Stored PDF '%s' as Document node", pdf_path)
            except Exception as e:
                logger.exception("❌ Failed to store PDF %s: %s", pdf_path, e)

        for url in urls:
            doc_id = self._safe_query_id(url)
            content = self._extract_text_from_url(url)
            if not content:
                logger.warning("URL %s has no extractable text, skipping.", url)
                continue
            try:
                with self.driver.session(database=self.database) as session:
                    session.execute_write(_tx, doc_id, f"URL:{url}", content)
                logger.info("💾 Stored URL '%s' as Document node", url)
            except Exception as e:
                logger.exception("❌ Failed to store URL %s: %s", url, e)

    # -------------------------
    # Core storage method
    # -------------------------
    def store_normalized_query(self, normalized_result: Dict[str, Any]) -> None:
        query_text = normalized_result.get("original_query", "")
        query_id = self._safe_query_id(query_text)
        entities = normalized_result.get("entities", [])

        def _tx(tx):
            tx.run(
                """
                MERGE (q:Query {id: $query_id})
                SET q.text = $query_text, q.is_medical = $is_medical, q.updated = datetime()
                """,
                query_id=query_id,
                query_text=query_text,
                is_medical=bool(normalized_result.get("is_medical_query", False)),
            )

            for ent in entities:
                cui = ent.get("cui")
                text = ent.get("text")
                label = ent.get("label")
                sem_types = ent.get("semantic_types") or []
                conf = float(ent.get("confidence", 1.0))
                if not cui:
                    continue

                tx.run(
                    """
                    MERGE (e:Entity {cui: $cui})
                    SET e.text = $text,
                        e.label = $label,
                        e.semantic_types = $sem_types,
                        e.last_seen = datetime()
                    """,
                    cui=cui, text=text, label=label, sem_types=sem_types
                )

                tx.run(
                    """
                    MATCH (q:Query {id: $query_id}), (e:Entity {cui: $cui})
                    MERGE (q)-[r:CONTAINS]->(e)
                    SET r.confidence = coalesce(r.confidence, 0.0) + $conf,
                        r.updated = datetime()
                    """,
                    query_id=query_id, cui=cui, conf=conf
                )

            # co-occurrence
            n = len(entities)
            for i in range(n):
                ent1 = entities[i]
                cui1 = ent1.get("cui")
                conf1 = float(ent1.get("confidence", 1.0))
                if not cui1:
                    continue
                for j in range(i+1, n):
                    ent2 = entities[j]
                    cui2 = ent2.get("cui")
                    conf2 = float(ent2.get("confidence", 1.0))
                    if not cui2:
                        continue
                    conf_min = min(conf1, conf2)
                    tx.run(
                        """
                        MATCH (a:Entity {cui: $cui1}), (b:Entity {cui: $cui2})
                        MERGE (a)-[r:CO_OCCURS_IN]->(b)
                        SET r.confidence = coalesce(r.confidence, 0.0) + $conf,
                            r.updated = datetime()
                        """,
                        cui1=cui1, cui2=cui2, conf=conf_min
                    )
                    tx.run(
                        """
                        MATCH (a:Entity {cui: $cui1}), (b:Entity {cui: $cui2})
                        MERGE (b)-[r2:CO_OCCURS_IN]->(a)
                        SET r2.confidence = coalesce(r2.confidence, 0.0) + $conf,
                            r2.updated = datetime()
                        """,
                        cui1=cui1, cui2=cui2, conf=conf_min
                    )

        try:
            with self.driver.session(database=self.database) as session:
                session.execute_write(_tx)
            logger.info("💾 Stored query '%s' with %d entities to Neo4j", query_id, len(entities))
        except Exception as e:
            logger.exception("❌ Failed to store normalized query: %s", e)
            raise

    # -------------------------
    # Subgraph query (fixed for Neo4j)
    # -------------------------
    def query_personalized_subgraph(self, cuis: List[str], max_depth: int = 2, limit: int = 50) -> List[Dict]:
        if not cuis:
            return []

        def _tx(tx):
            records = []
            for start_cui in cuis:
                query = f"""
                MATCH (start:Entity {{cui: '{start_cui}'}})
                MATCH path = (start)-[*1..{max_depth}]-(related:Entity)
                WHERE related.cui <> start.cui
                RETURN relationships(path) AS rels, related.cui AS target
                LIMIT {limit}
                """
                res = tx.run(query)
                for r in res:
                    rels = r["rels"] or []
                    rel_types = []
                    for rel in rels:
                        try:
                            rel_types.append(rel.type)
                        except Exception:
                            rel_types.append(str(rel))
                    records.append({"source": start_cui, "target": r["target"], "relationships": rel_types})
            return records

        try:
            with self.driver.session(database=self.database) as session:
                subgraph = session.execute_read(_tx)
            logger.info("📊 Retrieved subgraph with %d paths from Neo4j", len(subgraph))
            return subgraph
        except Exception as e:
            logger.exception("❌ Failed to query subgraph: %s", e)
            return []

    # -------------------------
    # PubMed integration (unchanged)
    # -------------------------
    def integrate_pubmed_evidence(self, query_text: str, cuis: List[str], normalized_result: Dict, max_per_cui: int = 3, pause_sec: float = 0.34) -> None:
        if not cuis:
            logger.info("No CUIs provided for PubMed integration.")
            return

        logger.info("🔬 Integrating real PubMed evidence for %d CUIs...", len(cuis))

        def _write_evidence(tx, cui: str, pmid: str, title: str, abstract: str, summary: str):
            tx.run(
                """
                MATCH (e:Entity {cui: $cui})
                MERGE (ev:Evidence {pubmed_id: $pmid})
                SET ev.title = $title,
                    ev.abstract = $abstract,
                    ev.summary = $summary,
                    ev.added = datetime()
                MERGE (e)-[:SUPPORTED_BY]->(ev)
                """,
                cui=cui,
                pmid=pmid,
                title=title,
                abstract=abstract,
                summary=summary,
            )

        for cui in cuis:
            try:
                entity_text = next((e.get("text") for e in normalized_result.get("entities", []) if e.get("cui") == cui), None)
                search_term = f"{entity_text or query_text}[Title/Abstract] AND (review[Publication Type] OR clinical trial[Publication Type])"
                handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_per_cui)
                search_results = Entrez.read(handle)
                handle.close()
                pmids = search_results.get("IdList", [])
                if not pmids:
                    time.sleep(pause_sec)
                    continue

                fetch_handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
                fetch_results = Entrez.read(fetch_handle)
                fetch_handle.close()

                pub_articles = fetch_results.get("PubmedArticle", [])
                with self.driver.session(database=self.database) as session:
                    for idx, pmid in enumerate(pmids[:max_per_cui]):
                        article = pub_articles[idx] if idx < len(pub_articles) else None
                        if not article:
                            continue
                        try:
                            title = article.get("MedlineCitation", {}).get("Article", {}).get("ArticleTitle", "")
                            abstract_obj = article.get("MedlineCitation", {}).get("Article", {}).get("Abstract", {})
                            if isinstance(abstract_obj, dict):
                                abs_text = abstract_obj.get("AbstractText", "")
                                if isinstance(abs_text, list):
                                    abstract = " ".join([str(a) for a in abs_text])
                                else:
                                    abstract = str(abs_text)
                            else:
                                abstract = str(abstract_obj or "")
                        except Exception:
                            title = ""
                            abstract = ""
                        summary = f"{title}: {abstract[:300]}..." if abstract else title
                        try:
                            session.execute_write(_write_evidence, cui, pmid, title, abstract, summary)
                        except Exception as e:
                            logger.exception("Failed to write evidence for pmid=%s cui=%s: %s", pmid, cui, e)
                time.sleep(pause_sec)
            except Exception as e:
                logger.exception("⚠️ PubMed error for CUI %s: %s", cui, e)
                time.sleep(pause_sec)
                continue

        logger.info("✅ Integrated real PubMed evidence for %d CUIs", len(cuis))


if __name__ == "__main__":
    handler = Neo4jHandler()

    # -------------------------
    # Add your PDF file paths here
    # -------------------------
    pdf_files = [
        r"C:\Users\DELL\Downloads\Harrisons-Neurology-in-Clinical-Medicine-2nd-Ed.pdf",
        r"C:\Users\DELL\Downloads\textbook-of-head-and-neck-anatomy.pdf"
    ]

    # -------------------------
    # Add your URLs here
    # -------------------------
    urls = [
        "https://seti.net/Neuron%20Lab/NeuronReferences/Neuroscience%20-%20Bear.pdf",
        "https://dn720006.ca.archive.org/0/items/betty-friedan-la-mistica-de-la-feminidad/The-Neuroscience-of-Intelligence.pdf",
        "https://www.assembleia.pr.leg.br/storage/pautas_comissoes/2025/02/1743442756_67ead344a9777.pdf"
    ]

    # Store PDFs and URLs in Neo4j
    handler.store_documents(pdf_paths=pdf_files, urls=urls)

    # Close connection
    handler.close()
