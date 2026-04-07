import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def validate_citations(driver, cited_ids):
    """
    Stage 3 — verify every paper ID exists in Neo4j before
    it reaches the LLM. Blocks hallucinated citations at source.
    """
    with driver.session() as session:
        result = session.run("""
            UNWIND $ids AS pid
            MATCH (p:Paper {id: pid})
            RETURN p.id AS id, p.title AS title
        """, ids=cited_ids)
        verified = {r["id"]: r["title"] for r in result}

    invalid = [pid for pid in cited_ids if pid not in verified]
    print(f"  Validated : {len(verified)}/{len(cited_ids)} citations exist in graph")
    if invalid:
        print(f"  Blocked   : {len(invalid)} hallucinated citations removed")

    return verified
