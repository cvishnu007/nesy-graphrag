import os
import sys
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    CLEAN_FILE, CHROMA_COLLECTION, CHROMA_DIR, DATA_SOURCE, EMBEDDING_MODEL, BATCH_SIZE
)

_collection = None
_embedder   = None


def get_collection():
    global _collection
    if _collection is None:
        client      = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def get_embedder():
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("Model loaded!")
    return _embedder


def build_index():
    """Run once — reads arxiv_clean.json, encodes, stores in ChromaDB."""
    df         = pd.read_json(CLEAN_FILE)
    collection = get_collection()
    embedder   = get_embedder()

    print(f"Loaded {len(df)} papers from {CLEAN_FILE}")
    print(f"ChromaDB collection '{CHROMA_COLLECTION}' ready — already stored: {collection.count()} papers")

    # resume support — skip already stored
    already_stored = set(collection.get()["ids"])
    df_remaining   = df[~df["id"].isin(already_stored)]
    print(f"Remaining to encode: {len(df_remaining)} papers")

    total = len(df_remaining)
    for i in range(0, total, BATCH_SIZE):
        batch      = df_remaining.iloc[i : i + BATCH_SIZE]
        embeddings = embedder.encode(
            batch["clean_abstract"].tolist(),
            show_progress_bar=False
        ).tolist()

        collection.add(
            ids        = batch["id"].tolist(),
            embeddings = embeddings,
            documents  = batch["clean_abstract"].tolist(),
            metadatas  = [
                {
                    "title"            : row["clean_title"],
                    "year"             : int(row["year"]),
                    "primary_category" : row["primary_category"],
                    "authors"          : ", ".join(row["authors"]) if isinstance(row["authors"], list) else "",
                    "doi"              : str(row["doi"]) if row["doi"] else "",
                    "paperId"          : str(row["paperId"]) if row.get("paperId") else str(row["id"]),
                    "corpusId"         : str(row["corpusId"]) if row.get("corpusId") else "",
                    "source_dataset"   : DATA_SOURCE
                }
                for _, row in batch.iterrows()
            ]
        )

        if (i + BATCH_SIZE) % 640 == 0 or i + BATCH_SIZE >= total:
            print(f"  Encoded {min(i + BATCH_SIZE, total)}/{total} papers...")

    print(f"\nDone! Total stored in ChromaDB: {collection.count()}")


def query(text, top_k=10):
    """Query ChromaDB with a text string, returns list of paper dicts."""
    collection = get_collection()
    embedder   = get_embedder()
    query_vec  = embedder.encode([text]).tolist()
    results    = collection.query(query_embeddings=query_vec, n_results=top_k)

    papers = []
    for i in range(len(results["ids"][0])):
        papers.append({
            "id"       : results["ids"][0][i],
            "abstract" : results["documents"][0][i],
            "title"    : results["metadatas"][0][i].get("title", ""),
            "year"     : results["metadatas"][0][i].get("year", ""),
            "category" : results["metadatas"][0][i].get("primary_category", ""),
            "score"    : 1.0,
            "source"   : "neural"
        })
    return papers


if __name__ == "__main__":
    os.makedirs(CHROMA_DIR, exist_ok=True)
    build_index()
