"""Select which concept graph the reasoning queries traverse.

``CONCEPT_SOURCE=spacy`` (default) uses the historical (:Paper)-[:RELATED_TO]->(:Concept)
noun-chunk graph. ``CONCEPT_SOURCE=scientific`` uses (:Paper)-[:MENTIONS]->(:SciEntity)
nodes produced by the SciERC-trained extractor (src/ingestion/scientific_ner.py).
"""
from src.utils.config import CONCEPT_SOURCE

CONCEPT_GRAPHS = {
    "spacy": ("Concept", "RELATED_TO"),
    "scientific": ("SciEntity", "MENTIONS"),
}
CONCEPT_LABEL, CONCEPT_RELATIONSHIP = CONCEPT_GRAPHS[CONCEPT_SOURCE]


def concept_cypher(query: str, source: str | None = None) -> str:
    """Rewrite a query written for the spaCy concept graph to the selected graph."""
    label, relationship = CONCEPT_GRAPHS[source or CONCEPT_SOURCE]
    return query.replace(":RELATED_TO", f":{relationship}").replace(":Concept", f":{label}")
