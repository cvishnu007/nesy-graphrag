"""Scientific entity extraction: SciERC evaluation and corpus extraction.

SciERC (Luan et al., EMNLP 2018) annotates AI/ML abstracts with six entity
types: Task, Method, Metric, Material, OtherScientificTerm, and Generic. This
module

* ``evaluate``: scores token-classification models trained on SciERC, and the
  historical spaCy noun-chunk extractor, on the held-out SciERC test set
  (exact-span typed F1, and untyped span F1 so spaCy is comparable);
* ``extract``: runs the selected model over the corpus abstracts and writes
  ``{id, entities: [{text, type}]}`` records. ``Generic`` mentions ("method",
  "approach") are dropped because they carry no content.

Output is a separate file; the historical spaCy NER file is never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

# These community SciERC checkpoints were uploaded without tokenizer files, so
# without this mapping every word becomes [UNK]. Load the tokenizer of the base
# model named in each checkpoint's config (vocabulary sizes match exactly).
TOKENIZER_FOR = {
    "anlp/SciERC_trained_Scibert": "allenai/scibert_scivocab_uncased",
    "roccas98/Scibert_finetuned_scierc": "allenai/scibert_scivocab_uncased",
    "anlp/SciERC_trained_SciDeBERTa": "KISTI-AI/scideberta-cs",
}
SCIERC_TYPES = ("Task","Method", "Metric", "Material", "OtherScientificTerm", "Generic")
_WORD = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")


def word_tokenize(text: str) -> list[str]:
    return _WORD.findall(text or "")


def bio_to_spans(labels: list[str]) -> list[tuple[int, int, str]]:
    """Decode word-level BIO labels into inclusive (start, end, type) spans."""
    spans, start, kind = [], None, None
    for index, label in enumerate(list(labels) + ["O"]):
        prefix, _, name = label.partition("-")
        if label == "O" or prefix == "B" or (prefix == "I" and name != kind):
            if start is not None:
                spans.append((start, index - 1, kind))
            start, kind = (index, name) if prefix in {"B", "I"} else (None, None)
    return spans


def load_scierc(path) -> list[dict]:
    """Load DyGIE-format SciERC: one record per sentence with gold spans."""
    sentences = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        offset = 0
        for tokens, entities in zip(doc["sentences"], doc["ner"]):
            gold = {(start - offset, end - offset, kind) for start, end, kind in entities}
            sentences.append({"doc": doc["doc_key"], "tokens": tokens, "gold": gold})
            offset += len(tokens)
    return sentences


class TokenClassifierNER:
    """Word-level predictions from a Hugging Face token-classification model."""

    def __init__(self, repo: str, device: str | None = None, batch_size: int = 32):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer_repo = TOKENIZER_FOR.get(repo, repo)
        if "deberta" in tokenizer_repo.lower() or "roberta" in tokenizer_repo.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, add_prefix_space=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
        self.model = AutoModelForTokenClassification.from_pretrained(repo).to(self.device).eval()
        if len(self.tokenizer) != self.model.config.vocab_size:
            raise ValueError(f"Tokenizer {tokenizer_repo} ({len(self.tokenizer)}) does not match "
                             f"model {repo} vocabulary ({self.model.config.vocab_size})")
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.batch_size = batch_size
        self._check_tokenizer()

    def _check_tokenizer(self):
        """Fail loudly if words map to [UNK] (a model uploaded without its vocabulary)."""
        words = "Graph neural networks improve node classification accuracy on citation datasets .".split()
        ids = self.tokenizer(words, is_split_into_words=True, add_special_tokens=False)["input_ids"]
        unknown = sum(token == self.tokenizer.unk_token_id for token in ids) / max(1, len(ids))
        if unknown > 0.2:
            raise ValueError(f"{unknown:.0%} of sample tokens are unknown; wrong tokenizer for this model")

    def predict(self, token_lists: list[list[str]]) -> list[list[tuple[int, int, str]]]:
        results = []
        with self.torch.inference_mode():
            for start in range(0, len(token_lists), self.batch_size):
                batch = token_lists[start:start + self.batch_size]
                encoded = self.tokenizer(batch, is_split_into_words=True, truncation=True,
                                         max_length=512, padding=True, return_tensors="pt")
                logits = self.model(**{k: v.to(self.device) for k, v in encoded.items()}).logits
                predictions = logits.argmax(-1).cpu().tolist()
                for row, tokens in enumerate(batch):
                    labels, seen = ["O"] * len(tokens), set()
                    for position, word in enumerate(encoded.word_ids(row)):
                        if word is None or word in seen:
                            continue
                        seen.add(word)  # first sub-token decides the word label
                        labels[word] = self.id2label[predictions[row][position]]
                    results.append(bio_to_spans(labels))
        return results


GLINER_LABELS = {
    "task": "Task", "method": "Method", "evaluation metric": "Metric",
    "dataset or material": "Material", "scientific term": "OtherScientificTerm",
}


class GlinerNER:
    """Zero-shot GLiNER with SciERC types described in plain language."""

    def __init__(self, repo: str = "urchade/gliner_medium-v2.1", threshold: float = 0.5):
        import torch
        from gliner import GLiNER

        from huggingface_hub import snapshot_download

        # Load from the cached folder: GLiNER's own loader demands every repo file,
        # including duplicate .bin weights that are not needed with safetensors.
        local_dir = repo if Path(repo).is_dir() else snapshot_download(
            repo, allow_patterns=["*.json", "*.txt", "model.safetensors"], local_files_only=True)
        self.model = GLiNER.from_pretrained(local_dir, local_files_only=True)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.threshold = threshold

    def predict(self, token_lists):
        results = []
        for tokens in token_lists:
            text, starts, cursor = "", [], 0
            for token in tokens:
                if text:
                    text += " "
                starts.append(len(text))
                text += token
            ends = [start + len(token) for start, token in zip(starts, tokens)]
            spans = set()
            for entity in self.model.predict_entities(text, list(GLINER_LABELS), threshold=self.threshold):
                first = next((i for i, s in enumerate(starts) if s >= entity["start"]), None)
                last = next((i for i in range(len(ends) - 1, -1, -1) if ends[i] <= entity["end"]), None)
                if first is not None and last is not None and first <= last:
                    spans.add((first, last, GLINER_LABELS[entity["label"]]))
            results.append(sorted(spans))
        return results


class SpacyChunkNER:
    """Historical extractor: spaCy entities plus noun chunks (untyped)."""

    def __init__(self):
        import spacy

        self.nlp = spacy.load("en_core_web_sm")

    def predict(self, token_lists):
        from spacy.tokens import Doc

        results = []
        for tokens in token_lists:
            doc = self.nlp(Doc(self.nlp.vocab, words=list(tokens)))
            spans = {(ent.start, ent.end - 1, "Entity") for ent in doc.ents
                     if ent.label_ in {"ORG", "PRODUCT", "GPE", "WORK_OF_ART", "EVENT"}}
            spans |= {(chunk.start, chunk.end - 1, "Entity") for chunk in doc.noun_chunks if len(chunk) <= 4}
            results.append(sorted(spans))
        return results


def prf(gold: set, predicted: set) -> dict:
    hits = len(gold & predicted)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "gold": len(gold), "predicted": len(predicted)}


def score_predictions(sentences: list[dict], predictions: list[list[tuple]], *, typed: bool,
                      exclude_types=()) -> dict:
    gold, predicted = set(), set()
    per_type = {kind: (set(), set()) for kind in SCIERC_TYPES if kind not in exclude_types}
    for index, (sentence, spans) in enumerate(zip(sentences, predictions)):
        spans = [span for span in spans if span[2] not in exclude_types]
        for start, end, kind in sentence["gold"]:
            if kind in exclude_types:
                continue
            gold.add((index, start, end, kind if typed else ""))
            per_type[kind][0].add((index, start, end))
        for start, end, kind in spans:
            predicted.add((index, start, end, kind if typed else ""))
            if kind in per_type:
                per_type[kind][1].add((index, start, end))
    report = prf(gold, predicted)
    if typed:
        report["per_type"] = {kind: prf(*sets) for kind, sets in per_type.items()}
    return report


def normalize_entity(text: str) -> str:
    """Lowercase, re-attach punctuation split by word tokenization, and drop a
    dangling abbreviation ("convolutional neural networks ( cnns" -> the long form)."""
    text = " ".join(text.lower().split())
    text = re.sub(r"\s+([)\],.;:!?])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s*\([^()]*\)?$", "", text)          # trailing "(abbr)" or unclosed "(abbr"
    text = re.sub(r"\s+-\s+", "-", text)
    return text.strip(" .,;:()[]")


def extract_corpus(model, papers, *, drop_types=("Generic",), sentence_batch=256):
    """Return {id, entities} records from original-case abstracts."""
    from src.pipeline.provenance import split_abstract_sentences

    units = []
    for paper in papers:
        abstract = paper.get("abstract") or paper.get("clean_abstract") or ""
        for sentence in split_abstract_sentences(abstract):
            tokens = word_tokenize(sentence)[:200]
            if tokens:
                units.append((paper["id"], tokens))
    entities = {paper["id"]: [] for paper in papers}
    report_every = max(1, len(units) // sentence_batch // 20)  # about 20 progress lines
    for batch_number, start in enumerate(range(0, len(units), sentence_batch), start=1):
        if batch_number % report_every == 0:
            print(f"NER progress: {start}/{len(units)} sentences ({start / len(units):.0%})", flush=True)
        batch = units[start:start + sentence_batch]
        for (paper_id, tokens), spans in zip(batch, model.predict([tokens for _, tokens in batch])):
            for begin, end, kind in spans:
                text = normalize_entity(" ".join(tokens[begin:end + 1]))
                if kind in drop_types or len(text) < 2 or not re.search(r"[a-z]", text):
                    continue
                entities[paper_id].append({"text": text, "type": kind})
    records = []
    for paper in papers:
        unique = {}
        for entity in entities[paper["id"]]:
            unique.setdefault(entity["text"], entity)  # first type wins for repeated mentions
        records.append({"id": paper["id"], "entities": list(unique.values())})
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--scierc", default="data/external/zj88zj__SCIERC/test.json")
    evaluate.add_argument("--models", nargs="+", required=True, help="HF repo IDs, or 'spacy'")
    evaluate.add_argument("--output", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--model", required=True)
    extract.add_argument("--output", default="data/s2_ner_scientific.json")
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        sentences = load_scierc(args.scierc)
        report = {"dataset": args.scierc, "sentences": len(sentences),
                  "gold_entities": sum(len(s["gold"]) for s in sentences),
                  "gold_types": dict(Counter(kind for s in sentences for *_, kind in s["gold"])), "models": {}}
        for name in args.models:
            if name == "spacy":
                model = SpacyChunkNER()
            elif name.startswith("gliner:"):
                model = GlinerNER(name.split(":", 1)[1])
            else:
                model = TokenClassifierNER(name)
            predictions = model.predict([s["tokens"] for s in sentences])
            report["models"][name] = {"untyped_span": score_predictions(sentences, predictions, typed=False)}
            if name != "spacy":
                report["models"][name]["typed"] = score_predictions(sentences, predictions, typed=True)
                report["models"][name]["typed_without_generic"] = score_predictions(
                    sentences, predictions, typed=True, exclude_types=("Generic",))
            print(name, json.dumps({k: {m: v[m] for m in ("precision", "recall", "f1")}
                                    for k, v in report["models"][name].items()}), flush=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        return 0

    from src.utils.config import CLEAN_FILE

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    papers = json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))
    records = extract_corpus(TokenClassifierNER(args.model, batch_size=64), papers)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"model": args.model, "dropped_types": ["Generic"], "records": records}),
                         encoding="utf-8")
    os.replace(temporary, output)
    counts = Counter(entity["type"] for record in records for entity in record["entities"])
    print(json.dumps({"papers": len(records), "entities": sum(counts.values()), "by_type": counts,
                      "empty_papers": sum(not r["entities"] for r in records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
