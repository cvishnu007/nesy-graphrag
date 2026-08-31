import math
import os
import sys
from functools import lru_cache
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    BATCH_SIZE, CHROMA_COLLECTION, CHROMA_DIR, CLEAN_FILE, DATA_SOURCE,
    EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL,
)
from src.utils.compute import configure_torch

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
        device = configure_torch()
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        local_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        try:
            _embedder = SentenceTransformer(
                EMBEDDING_MODEL,
                device=device,
                local_files_only=local_only,
            )
        except TypeError:
            _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print("Model loaded!")
    return _embedder


@lru_cache(maxsize=128)
def encode_query(text):
    """Encode and briefly cache query text shared by retrieval stages."""
    embedding = get_embedder().encode(
        [text],
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
    )[0]
    return tuple(float(value) for value in embedding)


def build_index():
    """Encode the configured cleaned dataset into the configured collection."""
    df         = pd.read_json(CLEAN_FILE)
    if df.empty:
        print(f"No papers found in {CLEAN_FILE}; nothing to index.")
        return
    required_columns = {"id", "clean_abstract", "clean_title", "year", "primary_category"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise RuntimeError(
            f"{CLEAN_FILE} is missing required columns: {', '.join(missing_columns)}"
        )
    collection = get_collection()
    embedder   = get_embedder()

    print(f"Loaded {len(df)} papers from {CLEAN_FILE}")
    print(f"ChromaDB collection '{CHROMA_COLLECTION}' ready; already stored: {collection.count()} papers")

    # resume support — skip already stored
    already_stored = set(collection.get()["ids"])
    df_remaining   = df[~df["id"].isin(already_stored)]
    print(f"Remaining to encode: {len(df_remaining)} papers")

    total = len(df_remaining)
    for i in range(0, total, BATCH_SIZE):
        batch      = df_remaining.iloc[i : i + BATCH_SIZE]
        embeddings = embedder.encode(
            batch["clean_abstract"].tolist(),
            batch_size=EMBEDDING_BATCH_SIZE,
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
    result_count = min(max(0, int(top_k)), collection.count())
    if result_count == 0:
        return []
    query_vec = [list(encode_query(text))]
    results    = collection.query(
        query_embeddings=query_vec,
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    papers = []
    for i in range(len(results["ids"][0])):
        distance = float(results["distances"][0][i])
        similarity = max(0.0, min(1.0, 1.0 - distance))
        papers.append({
            "id"       : results["ids"][0][i],
            "abstract" : results["documents"][0][i],
            "title"    : results["metadatas"][0][i].get("title", ""),
            "year"     : results["metadatas"][0][i].get("year", ""),
            "category" : results["metadatas"][0][i].get("primary_category", ""),
            "score"    : round(similarity, 6),
            "neural_score": round(similarity, 6),
            "source"   : "neural"
        })
    return papers


def score_papers_against_query(text, paper_ids):
    """Return cosine similarity between a query and stored paper embeddings.

    This reuses embeddings already stored in ChromaDB, so graph candidates do
    not need to be encoded again. IDs absent from the collection are omitted.
    """
    unique_ids = list(dict.fromkeys(paper_ids))
    if not text.strip() or not unique_ids:
        return {}

    stored = get_collection().get(ids=unique_ids, include=["embeddings"])
    stored_embeddings = stored.get("embeddings")
    if stored_embeddings is None or len(stored_embeddings) == 0:
        return {}

    query_embedding = encode_query(text)
    query_norm = math.sqrt(math.fsum(float(value) ** 2 for value in query_embedding))
    if query_norm == 0:
        return {}

    scores = {}
    for paper_id, embedding in zip(stored["ids"], stored_embeddings):
        paper_norm = math.sqrt(math.fsum(float(value) ** 2 for value in embedding))
        if paper_norm == 0:
            continue
        dot_product = math.fsum(
            float(left) * float(right)
            for left, right in zip(query_embedding, embedding)
        )
        similarity = dot_product / (query_norm * paper_norm)
        scores[paper_id] = round(max(0.0, min(1.0, similarity)), 6)
    return scores


if __name__ == "__main__":
    os.makedirs(CHROMA_DIR, exist_ok=True)
    build_index()
