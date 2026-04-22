import os
import sys
import pandas as pd
from neo4j import GraphDatabase

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    DATA_SOURCE,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    NER_FILE, NEO4J_BATCH_SIZE, MAX_AUTHORS, MAX_CONCEPTS, CITES_THRESHOLD,
    USE_REAL_CITATIONS
)


def get_driver():
    if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
        raise RuntimeError(
            "Missing Neo4j credentials. Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in .env."
        )
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Connected to Neo4j!")
    return driver


def drop_legacy_constraints(session):
    """Remove old constraints that conflict with the current data model."""
    constraints = session.run("""
        SHOW CONSTRAINTS
        YIELD name, type, labelsOrTypes, properties
        WHERE type = 'NODE_PROPERTY_UNIQUENESS'
          AND labelsOrTypes = ['Author']
          AND properties = ['name']
        RETURN name
    """)

    dropped = 0
    for record in constraints:
        constraint_name = record["name"]
        session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
        dropped += 1

    if dropped:
        print(f"Dropped {dropped} legacy Author.name constraint(s).")


def insert_papers(driver, df):
    """Insert Paper, Author, Concept nodes and relationships — from cell 06."""
    print("Clearing existing data...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("Cleared!")
        drop_legacy_constraints(session)
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.key IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")

    print(f"Inserting {len(df)} papers in batches of {NEO4J_BATCH_SIZE}...")
    for i in range(0, len(df), NEO4J_BATCH_SIZE):
        batch = df.iloc[i:i + NEO4J_BATCH_SIZE]

        papers_data = [
            {
                "id"              : row["id"],
                "paperId"         : row["paperId"] if row.get("paperId") else row["id"],
                "corpusId"        : int(row["corpusId"]) if row.get("corpusId") and str(row.get("corpusId")).isdigit() else None,
                "title"           : row["clean_title"],
                "year"            : int(row["year"]) if pd.notna(row["year"]) else 0,
                "category"        : row["primary_category"],
                "abstract"        : row["clean_abstract"],
                "doi"             : str(row.get("doi")) if row.get("doi") else "",
                "venue"           : str(row.get("venue")) if row.get("venue") else "",
                "source"          : str(row.get("source")) if row.get("source") else DATA_SOURCE,
                "citationCount"   : int(row.get("citationCount") or 0),
                "referenceCount"  : int(row.get("referenceCount") or 0),
                "authors"         : _build_authors(row),
                "concepts"        : [c.lower().strip() for c in row["entities"][:MAX_CONCEPTS] if len(c) > 3]
                                    if isinstance(row["entities"], list) else [],
                "references"      : _normalize_refs(row.get("references"))
            }
            for _, row in batch.iterrows()
        ]

        with driver.session() as session:
            # Papers + Authors
            session.run("""
                UNWIND $papers AS p
                MERGE (paper:Paper {id: p.id})
                SET paper.title = p.title, paper.year = p.year,
                    paper.category = p.category, paper.abstract = p.abstract,
                    paper.paperId = p.paperId, paper.corpusId = p.corpusId,
                    paper.doi = p.doi, paper.venue = p.venue, paper.source = p.source,
                    paper.citationCount = p.citationCount, paper.referenceCount = p.referenceCount
                WITH paper, p
                UNWIND p.authors AS author
                MERGE (a:Author {key: author.key})
                SET a.name = author.name,
                    a.authorId = CASE WHEN author.authorId = '' THEN a.authorId ELSE author.authorId END
                MERGE (paper)-[:AUTHORED_BY]->(a)
            """, papers=papers_data)

            # Papers + Concepts
            session.run("""
                UNWIND $papers AS p
                MERGE (paper:Paper {id: p.id})
                WITH paper, p
                UNWIND p.concepts AS conceptName
                MERGE (c:Concept {name: conceptName})
                MERGE (paper)-[:RELATED_TO]->(c)
            """, papers=papers_data)

        print(f"  Inserted {min(i + NEO4J_BATCH_SIZE, len(df))}/{len(df)} papers...")

    # Verify
    with driver.session() as session:
        papers   = session.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
        authors  = session.run("MATCH (a:Author) RETURN count(a) AS c").single()["c"]
        concepts = session.run("MATCH (c:Concept) RETURN count(c) AS c").single()["c"]
        rels     = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

    print(f"\nDone! Papers: {papers} | Authors: {authors} | Concepts: {concepts} | Relations: {rels}")


def _build_authors(row):
    names = row.get("authors")
    ids = row.get("author_ids")
    names = names if isinstance(names, list) else []
    ids = ids if isinstance(ids, list) else []

    authors = []
    for idx, raw_name in enumerate(names[:MAX_AUTHORS]):
        name = str(raw_name).lower().strip()
        if not name:
            continue
        aid = str(ids[idx]).strip() if idx < len(ids) and ids[idx] else ""
        key = f"s2:{aid}" if aid else f"name:{name}"
        authors.append({"key": key, "name": name, "authorId": aid})
    return authors


def _normalize_refs(refs):
    if not isinstance(refs, list):
        return []
    cleaned = []
    seen = set()
    for r in refs:
        rid = str(r).strip()
        if not rid or rid in seen:
            continue
        seen.add(rid)
        cleaned.append(rid)
    return cleaned


def create_real_cites_edges(driver, df):
    """Create real CITES edges from references if available in data."""
    print("Creating CITES edges from dataset references...")
    edges_created_total = 0

    for i in range(0, len(df), NEO4J_BATCH_SIZE):
        batch = df.iloc[i:i + NEO4J_BATCH_SIZE]
        papers_data = []
        for _, row in batch.iterrows():
            refs = _normalize_refs(row.get("references"))
            if refs:
                papers_data.append({"id": row["id"], "references": refs})

        if not papers_data:
            continue

        with driver.session() as session:
            result = session.run("""
                UNWIND $papers AS p
                MATCH (source:Paper {id: p.id})
                UNWIND p.references AS rid
                MATCH (target:Paper {id: rid})
                MERGE (source)-[r:CITES]->(target)
                SET r.simulated = false, r.source = "semantic_scholar"
                RETURN count(r) AS edges_created
            """, papers=papers_data)
            edges_created_total += result.single()["edges_created"]

    print(f"Real CITES edges created: {edges_created_total}")
    return edges_created_total


def create_simulated_cites_edges(driver):
    """Create proxy CITES edges from shared concepts (fallback mode)."""
    print(f"Creating simulated CITES edges with threshold = {CITES_THRESHOLD}...")
    with driver.session() as session:
        result = session.run("""
            MATCH (p1:Paper)-[:RELATED_TO]->(c:Concept)<-[:RELATED_TO]-(p2:Paper)
            WHERE p1.id < p2.id
            WITH p1, p2, count(c) AS shared
            WHERE shared >= $threshold
            MERGE (p1)-[r:CITES]->(p2)
            SET r.shared_concepts = shared, r.simulated = true, r.source = "concept_overlap"
            RETURN count(*) AS edges_created
        """, threshold=CITES_THRESHOLD)
        edges = result.single()["edges_created"]
        print(f"Simulated CITES edges created: {edges}")
        return edges


def create_cites_edges(driver, df, use_real_citations=False):
    """Create CITES edges using real references first, then fallback if needed."""
    print("Clearing old CITES edges...")
    with driver.session() as session:
        session.run("MATCH ()-[r:CITES]->() DELETE r")
        print("Old CITES cleared!")

    if use_real_citations:
        real_edges = create_real_cites_edges(driver, df)
        if real_edges > 0:
            return
        print("No real reference edges found in dataset. Falling back to simulated CITES.")

    create_simulated_cites_edges(driver)


if __name__ == "__main__":
    driver = get_driver()

    df = pd.read_json(NER_FILE)
    print(f"Loaded {len(df)} papers from {NER_FILE}")

    insert_papers(driver, df)
    create_cites_edges(driver, df, use_real_citations=USE_REAL_CITATIONS)

    driver.close()
    print("\nNeo4j setup complete!")
