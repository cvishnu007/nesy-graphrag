import os
import sys
import pandas as pd
import spacy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import CLEAN_FILE, NER_FILE

# ── Noise words to filter out — from your notebook ────
NOISE = {
    "which", "order", "the", "a", "an", "this", "that", "these", "those",
    "we", "our", "their", "its", "it", "one", "two", "three", "kind",
    "way", "fact", "case", "result", "problem", "approach", "method",
    "the first one", "the second one", "a kind", "a single"
}


def extract_entities(nlp, text):
    if not text:
        return []
    doc = nlp(text[:1000])
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "GPE", "WORK_OF_ART", "EVENT"]:
            entities.append(ent.text.lower().strip())
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 4:
            entities.append(chunk.text.lower().strip())
    return list(set(entities))


def filter_entities(entities):
    cleaned = []
    for e in entities:
        if e in NOISE:
            continue
        if len(e.split()) == 1 and len(e) < 4:
            continue
        if e.startswith(("the ", "a ", "an ", "our ", "their ")):
            continue
        cleaned.append(e)
    return cleaned


def run():
    df = pd.read_json(CLEAN_FILE)
    print(f"Loaded {len(df)} papers from {CLEAN_FILE}")

    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded!")

    print("Extracting entities...")
    df["entities"] = df["clean_abstract"].apply(lambda t: extract_entities(nlp, t))

    print("Filtering entities...")
    df["entities"] = df["entities"].apply(filter_entities)

    print(f"\nSample entities from paper 0:\n{df['entities'].iloc[0][:10]}")
    print(f"\nSample entities from paper 1:\n{df['entities'].iloc[1][:10]}")

    df.to_json(NER_FILE, orient="records", indent=2)
    print(f"\nDone! Saved → {NER_FILE}")
    print(f"Papers with entities: {(df['entities'].str.len() > 0).sum()}")


if __name__ == "__main__":
    run()
