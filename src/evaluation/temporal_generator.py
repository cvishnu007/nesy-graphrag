"""Time-sliced test of the hypothesis *generator* (not only its signals).

For each task T from the time-sliced benchmark (``temporal_hypothesis.py``:
pre-cutoff document frequency >= 10, >= 3 confirmed methods), the task name is
the query, and four arms each propose up to 10 "apply method M to task T"
hypotheses using only pre-cutoff papers:

* ``bridge``: a pre-cutoff port of ``src/pipeline/hypothesis_bridge.py`` (seeds =
  top 5 papers by BGE query similarity; neighbours = top 15; candidates share a
  specific Task/Material entity with the seeds, lie outside the neighbours, are
  not citation-linked to the seeds, and pass the relevance floor (weakest seed
  minus 0.10). Their Methods unseen among the neighbours are proposed, candidate
  by candidate in bridge-score order;
* ``vector_neighbour``: Methods of vector ranks 6+ in order (the human-study baseline);
* ``popularity``: the most frequent pre-cutoff Methods;
* ``random``: random vocabulary Methods.

Every arm proposes only Methods that never co-occur with T before the cutoff. A
proposal is confirmed if (T, M) co-occurs in >= 2 papers after the cutoff.
Metric: precision@10 per task (unfilled slots count as misses), plus the share of
tasks with >= 1 confirmed proposal. Predeclared 2026-11-15; nothing tuned.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path("results/reasoning/temporal_generator_20261115")


def propose(ordered_methods, known, limit=10):
    out = []
    for method in ordered_methods:
        if method not in known and method not in out:
            out.append(method)
            if len(out) == limit:
                break
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoff", type=int, default=2022)
    parser.add_argument("--min-future", type=int, default=2)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    from src.evaluation.contradiction_index import PLAN, citation_map
    from src.evaluation.significance import paired_permutation_test
    from src.evaluation.temporal_hypothesis import load_mentions
    from src.pipeline.hypothesis_bridge import bridge_score, is_specific
    from src.storage.chroma_store import encode_query, get_collection
    from src.utils.config import CLEAN_FILE

    corpus = json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))
    year = {p["id"]: int(p.get("year") or 0) for p in corpus}
    past = sorted(pid for pid, y in year.items() if 0 < y <= args.cutoff)
    past_set = set(past)
    types, entities = defaultdict(Counter), defaultdict(set)
    for pid, name, kind in load_mentions():
        entities[pid].add(name)
        types[name][kind] += 1
    etype = {n: c.most_common(1)[0][0] for n, c in types.items()}
    df = Counter(e for pid in past for e in entities.get(pid, ()))
    max_df = 0.01 * len(past)
    vocab = {e for e, n in df.items() if 3 <= n <= max_df and is_specific(e) and etype[e] in ("Task", "Method")}
    problem_vocab = {e for e, n in df.items() if 2 <= n <= 500 and is_specific(e) and etype[e] in ("Task", "Material")}
    methods_vocab = sorted(e for e in vocab if etype[e] == "Method")
    past_entities = {pid: entities.get(pid, set()) for pid in past}

    known = defaultdict(set)
    for pid in past:
        ents = past_entities[pid]
        for t in ents:
            if t in vocab and etype[t] == "Task":
                known[t] |= {m for m in ents if m in vocab and etype[m] == "Method"}
    future = Counter()
    for pid, y in year.items():
        if y > args.cutoff:
            ents = entities.get(pid, set())
            for t in ents:
                if t in vocab and etype[t] == "Task":
                    for m in ents:
                        if m in vocab and etype[m] == "Method" and m not in known[t]:
                            future[(t, m)] += 1
    confirmed = {pair for pair, n in future.items() if n >= args.min_future}
    tasks = sorted(t for t in vocab if etype[t] == "Task" and df[t] >= 10
                   and sum((t, m) in confirmed for m in methods_vocab) >= 3)
    print(f"past papers {len(past)}; eligible tasks {len(tasks)}", flush=True)

    cites = citation_map(corpus, json.loads(PLAN.read_text()))
    linked = defaultdict(set)
    for source, targets in cites.items():
        if source in past_set:
            for target in targets:
                if target in past_set:
                    linked[source].add(target)
                    linked[target].add(source)
    stored = get_collection().get(ids=past, include=["embeddings"])
    ids = list(stored["ids"])
    E = np.asarray(stored["embeddings"], dtype=np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    mentions_by_problem = defaultdict(set)
    for pid in ids:
        for e in past_entities.get(pid, ()):
            if e in problem_vocab:
                mentions_by_problem[e].add(pid)
    position = {pid: i for i, pid in enumerate(ids)}
    popularity = sorted(methods_vocab, key=lambda m: (-df[m], m))
    rng = np.random.default_rng(20261115)

    per_arm = defaultdict(dict)
    examples = {}
    for task in tasks:
        q = np.asarray(encode_query(task), dtype=np.float32)
        q /= max(np.linalg.norm(q), 1e-12)
        sims = E @ q
        order = np.argsort(-sims)
        seeds = [ids[i] for i in order[:5]]
        neighbours = [ids[i] for i in order[:15]]
        floor = float(min(sims[order[:5]])) - 0.10
        neighbour_entities = set().union(*(past_entities.get(n, set()) for n in neighbours))
        seed_problems = {e for s in seeds for e in past_entities.get(s, ()) if e in problem_vocab}
        candidates = set().union(*(mentions_by_problem[e] for e in seed_problems)) if seed_problems else set()
        excluded = set(neighbours) | set().union(*(linked[s] for s in seeds))
        scored = []
        for c in candidates - excluded:
            sim = float(sims[position[c]])
            if sim < floor:
                continue
            problems = [e for e in past_entities[c] if e in seed_problems]
            novel = sorted((m for m in past_entities[c] if m in vocab and etype[m] == "Method"
                            and m not in neighbour_entities and df[m] >= 2), key=lambda m: df[m])
            if problems and novel:
                scored.append((bridge_score([df[e] for e in problems], [df[m] for m in novel], sim, len(past)), c, novel))
        scored.sort(key=lambda x: (-x[0], x[1]))
        bridge = propose([m for _, _, novel in scored for m in novel], known[task], args.limit)
        vector = propose([m for i in order[5:300] for m in sorted(past_entities.get(ids[i], ()), key=lambda e: df.get(e, 0))
                          if m in vocab and etype[m] == "Method"], known[task], args.limit)
        arms = {"bridge": bridge, "vector_neighbour": vector,
                "popularity": propose(popularity, known[task], args.limit),
                "random": propose(list(rng.permutation(methods_vocab)), known[task], args.limit)}
        for arm, proposals in arms.items():
            hits = sum((task, m) in confirmed for m in proposals)
            per_arm[arm][task] = {"precision@10": hits / args.limit, "hits": hits, "proposed": len(proposals)}
        if len(examples) < 15 and any((task, m) in confirmed for m in bridge):
            examples[task] = [{"method": m, "confirmed": (task, m) in confirmed, "future_papers": future.get((task, m), 0)}
                              for m in bridge]

    summary = {arm: {"precision@10": round(float(np.mean([v[t]["precision@10"] for t in tasks])), 4),
                     "tasks_with_hit": sum(v[t]["hits"] > 0 for t in tasks),
                     "mean_proposed": round(float(np.mean([v[t]["proposed"] for t in tasks])), 2)}
               for arm, v in per_arm.items()}
    tests = {}
    for a, b in (("bridge", "vector_neighbour"), ("bridge", "popularity"), ("bridge", "random"),
                 ("vector_neighbour", "random")):
        deltas = [per_arm[a][t]["precision@10"] - per_arm[b][t]["precision@10"] for t in tasks]
        r = paired_permutation_test(deltas)
        tests[f"{a}_vs_{b}"] = {"mean_delta": round(r["mean_delta"], 4),
                                "bootstrap_95_ci": [round(x, 4) for x in r["bootstrap_95_ci"]],
                                "p": r["monte_carlo_two_sided_p"]}
    report = {"settings": vars(args), "tasks": len(tasks), "arms": summary, "comparisons": tests,
              "bridge_examples": examples}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "bridge_examples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
