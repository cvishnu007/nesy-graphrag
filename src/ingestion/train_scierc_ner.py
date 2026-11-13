"""Fine-tune SciBERT for SciERC entity recognition (train -> select on dev -> test once).

Model training was reopened by the project owner on 2026-11-14 for this NER
component only. Standard protocol: the epoch with the best dev typed F1 is kept,
and the test split is scored once with that checkpoint. SciERC contains some
nested entities; a flat BIO tagger keeps the longest span when spans overlap,
so nested inner spans count as misses in test scoring (as in flat-NER baselines).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from src.ingestion.scientific_ner import load_scierc, score_predictions

ENTITY_TYPES = ("Task", "Method", "Metric", "Material", "OtherScientificTerm", "Generic")
LABELS = ["O"] + [f"{prefix}-{kind}" for kind in ENTITY_TYPES for prefix in ("B", "I")]


def flat_bio(tokens, spans):
    """Longest-first flat BIO labels; overlapping shorter spans are dropped."""
    labels = ["O"] * len(tokens)
    for start, end, kind in sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0])):
        if all(labels[i] == "O" for i in range(start, end + 1)):
            labels[start] = f"B-{kind}"
            for i in range(start + 1, end + 1):
                labels[i] = f"I-{kind}"
    return labels


def encode(tokenizer, sentences, max_length=256):
    label_ids = {label: i for i, label in enumerate(LABELS)}
    rows = []
    for sentence in sentences:
        words = sentence["tokens"]
        tags = flat_bio(words, sentence["gold"])
        encoded = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length)
        targets, previous = [], None
        for word in encoded.word_ids():
            targets.append(-100 if word is None or word == previous else label_ids[tags[word]])
            previous = word
        rows.append({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": targets})
    return rows


def batches(rows, size, pad_id, shuffle):
    import torch

    order = list(range(len(rows)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), size):
        chunk = [rows[i] for i in order[start:start + size]]
        width = max(len(r["input_ids"]) for r in chunk)
        yield {
            "input_ids": torch.tensor([r["input_ids"] + [pad_id] * (width - len(r["input_ids"])) for r in chunk]),
            "attention_mask": torch.tensor([r["attention_mask"] + [0] * (width - len(r["attention_mask"])) for r in chunk]),
            "labels": torch.tensor([r["labels"] + [-100] * (width - len(r["labels"])) for r in chunk]),
        }


class _InMemoryNER:
    """Adapter so the training loop can reuse TokenClassifierNER.predict."""

    def __init__(self, model, tokenizer, device):
        from src.ingestion.scientific_ner import TokenClassifierNER
        import torch

        self.predictor = TokenClassifierNER.__new__(TokenClassifierNER)
        self.predictor.torch, self.predictor.device = torch, device
        self.predictor.tokenizer, self.predictor.model = tokenizer, model
        self.predictor.id2label = dict(enumerate(LABELS))
        self.predictor.batch_size = 64

    def predict(self, token_lists):
        return self.predictor.predict(token_lists)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/external/zj88zj__SCIERC")
    parser.add_argument("--base", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--output-dir", default="models/scibert-scierc-ner")
    parser.add_argument("--report", default="results/external/scierc_ner_finetune_20261114.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {split: load_scierc(Path(args.data_dir) / f"{split}.json") for split in ("train", "dev", "test")}
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)}).to(device)
    train_rows = encode(tokenizer, data["train"])
    steps = args.epochs * ((len(train_rows) + args.batch_size - 1) // args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    predictor = _InMemoryNER(model, tokenizer, device)
    output = Path(args.output_dir)
    history, best = [], -1.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in batches(train_rows, args.batch_size, tokenizer.pad_token_id, shuffle=True):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
        model.eval()
        dev = score_predictions(data["dev"], predictor.predict([s["tokens"] for s in data["dev"]]), typed=True)
        history.append({"epoch": epoch, "train_loss": round(total, 3), "dev_typed_f1": dev["f1"]})
        print(json.dumps(history[-1]), flush=True)
        if dev["f1"] > best:
            best = dev["f1"]
            output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
    best_model = AutoModelForTokenClassification.from_pretrained(output).to(device).eval()
    test_predictions = _InMemoryNER(best_model, tokenizer, device).predict([s["tokens"] for s in data["test"]])
    report = {
        "base": args.base, "seed": args.seed, "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "train_sentences": len(data["train"]), "dev_sentences": len(data["dev"]), "test_sentences": len(data["test"]),
        "selection": "best dev typed exact-span F1; test scored once", "history": history,
        "best_dev_typed_f1": best, "training_seconds": round(time.time() - started),
        "test": {"typed": score_predictions(data["test"], test_predictions, typed=True),
                 "untyped_span": score_predictions(data["test"], test_predictions, typed=False)},
        "model_dir": str(output),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print("TEST", json.dumps({k: {m: v[m] for m in ("precision", "recall", "f1")} for k, v in report["test"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
