import os
import sys
import pandas as pd
import spacy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import CLEAN_FILE, NER_FILE
from src.utils.compute import available_cpu_workers

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


def _atomic_write_frame(df, path):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    df.to_json(temporary, orient="records", indent=2)
    os.replace(temporary, path)


def run():
    df = pd.read_json(CLEAN_FILE)
    print(f"Loaded {len(df)} papers from {CLEAN_FILE}")
    if df.empty:
        df["entities"] = pd.Series(dtype=object)
        _atomic_write_frame(df, NER_FILE)
        print(f"No papers to process. Saved empty NER dataset to {NER_FILE}")
        return
    if "clean_abstract" not in df.columns:
        raise RuntimeError(f"{CLEAN_FILE} does not contain a clean_abstract column")
    if "id" not in df.columns:
        raise RuntimeError(f"{CLEAN_FILE} does not contain an id column")

    existing_entities = {}
    if os.path.exists(NER_FILE):
        existing_df = pd.read_json(NER_FILE)
        if {"id", "entities"}.issubset(existing_df.columns):
            existing_entities = {
                row["id"]: row["entities"]
                for _, row in existing_df.iterrows()
                if isinstance(row["entities"], list)
            }
    df["entities"] = [existing_entities.get(paper_id) for paper_id in df["id"]]
    remaining_indices = [
        index
        for index, entities in df["entities"].items()
        if not isinstance(entities, list)
    ]
    print(
        f"NER resume: {len(df) - len(remaining_indices)} reused, "
        f"{len(remaining_indices)} remaining"
    )
    if not remaining_indices:
        _atomic_write_frame(df, NER_FILE)
        print(f"All {len(df)} papers already have NER output.")
        return

    spacy_device = os.getenv("SPACY_DEVICE", "auto").strip().lower()
    if spacy_device not in {"auto", "gpu", "cpu"}:
        raise ValueError("SPACY_DEVICE must be auto, gpu, or cpu")
    if spacy_device == "gpu":
        spacy.require_gpu(int(os.getenv("SPACY_GPU_ID", "0")))
        gpu_enabled = True
    elif spacy_device == "auto":
        gpu_enabled = spacy.prefer_gpu(int(os.getenv("SPACY_GPU_ID", "0")))
    else:
        spacy.require_cpu()
        gpu_enabled = False

    nlp = spacy.load("en_core_web_sm")
    print(f"spaCy model loaded on {'gpu' if gpu_enabled else 'cpu'}!")

    print("Extracting entities...")
    n_process = 1 if gpu_enabled else available_cpu_workers("SPACY_N_PROCESS")
    batch_size = max(1, int(os.getenv("SPACY_BATCH_SIZE", "128")))
    checkpoint_size = max(1, int(os.getenv("NER_CHECKPOINT_SIZE", "5000")))
    print(
        f"spaCy workers: n_process={n_process}, batch_size={batch_size}, "
        f"checkpoint_size={checkpoint_size}"
    )

    def extract_from_doc(doc):
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "GPE", "WORK_OF_ART", "EVENT"]:
                entities.append(ent.text.lower().strip())
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:
                entities.append(chunk.text.lower().strip())
        return list(set(entities))

    processed = 0
    for start in range(0, len(remaining_indices), checkpoint_size):
        indices = remaining_indices[start:start + checkpoint_size]
        texts = [
            str(df.at[index, "clean_abstract"])[:1000]
            if df.at[index, "clean_abstract"]
            else ""
            for index in indices
        ]
        extracted = [
            filter_entities(extract_from_doc(doc))
            for doc in nlp.pipe(
                texts,
                batch_size=batch_size,
                n_process=n_process,
            )
        ]
        for index, entities in zip(indices, extracted):
            df.at[index, "entities"] = entities

        processed += len(indices)
        _atomic_write_frame(df, NER_FILE)
        print(
            f"NER checkpoint: {processed}/{len(remaining_indices)} new papers; "
            f"{len(df) - len(remaining_indices) + processed}/{len(df)} total ready"
        )

    for index in range(min(2, len(df))):
        print(f"\nSample entities from paper {index}:\n{df['entities'].iloc[index][:10]}")

    _atomic_write_frame(df, NER_FILE)
    print(f"\nDone! Saved to {NER_FILE}")
    print(f"Papers with entities: {(df['entities'].str.len() > 0).sum()}")


if __name__ == "__main__":
    run()
