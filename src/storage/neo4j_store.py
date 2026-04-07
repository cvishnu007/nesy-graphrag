import os
import sys
import pandas as pd
from neo4j import GraphDatabase

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    NER_FILE, NEO4J_BATCH_SIZE, MAX_AUTHORS, MAX_CONCEPTS, CITES_THRESHOLD
)


def get_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Connected to Neo4j!")
    return driver


def insert_papers(driver, df):
    """Insert Paper, Author, Concept nodes and relationships — from cell 06."""
    print("Clearing existing data...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("Cleared!")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")

    print(f"Inserting {len(df)} papers in batches of {NEO4J_BATCH_SIZE}...")
    for i in range(0, len(df), NEO4J_BATCH_SIZE):
        batch = df.iloc[i:i + NEO4J_BATCH_SIZE]

        papers_data = [
            {
                "id"       : row["id"],
                "title"    : row["clean_title"],
                "year"     : int(row["year"]),
                "category" : row["primary_category"],
                "abstract" : row["clean_abstract"],
                "authors"  : [a.lower().strip() for a in row["authors"][:MAX_AUTHORS]]
                              if isinstance(row["authors"], list) else [],
                "concepts" : [c.lower().strip() for c in row["entities"][:MAX_CONCEPTS] if len(c) > 3]
                              if isinstance(row["entities"], list) else []
            }
            for _, row in batch.iterrows()
        ]

        with driver.session() as session:
            # Papers + Authors
            session.run("""
                UNWIND $papers AS p
                MERGE (paper:Paper {id: p.id})
                SET paper.title = p.title, paper.year = p.year,
                    paper.category = p.category, paper.abstract = p.abstract
                WITH paper, p
                UNWIND p.authors AS authorName
                MERGE (a:Author {name: authorName})
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


def create_cites_edges(driver):
    """Create CITES edges from shared concepts — from cell 07."""
    print("Clearing old CITES edges...")
    with driver.session() as session:
        session.run("MATCH ()-[r:CITES]->() DELETE r")
        print("Old CITES cleared!")

    print(f"Creating CITES edges with threshold = {CITES_THRESHOLD}...")
    with driver.session() as session:
        result = session.run("""
            MATCH (p1:Paper)-[:RELATED_TO]->(c:Concept)<-[:RELATED_TO]-(p2:Paper)
            WHERE p1.id < p2.id
            WITH p1, p2, count(c) AS shared
            WHERE shared >= $threshold
            MERGE (p1)-[:CITES {shared_concepts: shared, simulated: true}]->(p2)
            RETURN count(*) AS edges_created
        """, threshold=CITES_THRESHOLD)
        edges = result.single()["edges_created"]
        print(f"CITES edges created: {edges}")


if __name__ == "__main__":
    driver = get_driver()

    df = pd.read_json(NER_FILE)
    print(f"Loaded {len(df)} papers from {NER_FILE}")

    insert_papers(driver, df)
    create_cites_edges(driver)

    driver.close()
    print("\nNeo4j setup complete!")
