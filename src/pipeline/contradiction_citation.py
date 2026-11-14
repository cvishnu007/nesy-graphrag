"""Citation-grounded contradiction candidates.

Human labels on 68 NLI-screened pairs found no real contradictions: topic-similar
finding sentences mostly pair two papers that each report their own good
results. In CS literature real disagreements are usually written on purpose
(replications, "revisiting X", negative results, comments), and such a paper
cites the work it disputes. This generator therefore:

1. takes each retrieved paper and its citation neighbours (``CITES`` and, when
   present, ``CITES_ADDED``), so every pair is linked by a citation;
2. keeps *critical* sentences in the citing paper: findings that use contrastive
   language ("contrary to", "fails to", "does not generalize", "we revisit");
3. pairs each critical sentence with the cited paper's sentences that name the
   same specific scientific entity (SciERC ``SciEntity``, generic names excluded),
   so both sentences are about the same subject;
4. ranks by symmetric NLI contradiction probability.

As with ``contradiction_nli``, the score only screens candidates; a verdict model
decides.
"""
from __future__ import annotations

import re

from src.pipeline.contradiction_nli import contradiction_probabilities, is_finding
from src.pipeline.provenance import passage_id, split_abstract_sentences

CRITICAL = re.compile(
    r"\b(contrary to|in contrast to (prior|previous|existing|earlier|common|popular)|"
    r"fail(s|ed)? to|does not (hold|generalize|improve|outperform|scale|help)|"
    r"do not (hold|generalize|improve|outperform|scale|help)|did not (improve|outperform|help)|"
    r"no (significant|consistent|clear) (improvement|gain|benefit|difference|advantage)|"
    r"overestimat\w*|underestimat\w*|underperform\w*|not (as |more )?effective|"
    # "reproducibility"/"reproducible" mostly mark tooling papers, not disagreements.
    r"revisit\w*|re-?examin\w*|re-?evaluat\w*|reproduc(e|ed|ing|tion)|replicat\w*|"
    r"we (question|challenge|refute|dispute|show that .{0,40}\bnot\b)|"
    r"misleading|flawed|unreliable|(is|are) not (necessary|needed|sufficient|robust)|"
    r"surprising(ly)?|counter-?intuitive|simple baselines?|on par with)\b",
    re.IGNORECASE,
)
EDGES = "CITES|CITES_ADDED"


def critical_sentences(paper):
    rows = []
    for index, sentence in enumerate(split_abstract_sentences(paper.get("abstract") or ""), start=1):
        if is_finding(sentence) and CRITICAL.search(sentence):
            rows.append({"id": passage_id(paper["id"], index), "paper_id": paper["id"], "sentence_index": index,
                         "text": sentence})
    return rows


def all_sentences(paper):
    return [{"id": passage_id(paper["id"], i), "paper_id": paper["id"], "sentence_index": i, "text": s}
            for i, s in enumerate(split_abstract_sentences(paper.get("abstract") or ""), start=1)
            if len(s.split()) >= 6]


def shared_subjects(first_text, second_text, entities):
    a, b = first_text.lower(), second_text.lower()
    return sorted(e for e in entities if len(e) > 3 and e in a and e in b)


def citation_neighbours(driver, paper_ids, *, per_paper=40):
    """Return (citing, cited) pairs touching the given papers, with both papers' text and entities."""
    query = f"""
        UNWIND $ids AS pid
        MATCH (p:Paper {{id: pid}})-[r:{EDGES}]-(q:Paper)
        WITH p, q, startNode(r) = p AS p_cites
        ORDER BY q.id
        WITH p, collect({{q: q, p_cites: p_cites}})[..$per_paper] AS links
        UNWIND links AS link
        WITH p, link.q AS q, link.p_cites AS p_cites
        OPTIONAL MATCH (p)-[:MENTIONS]->(e:SciEntity)<-[:MENTIONS]-(q)
        WHERE NOT coalesce(e.generic, false)
        RETURN CASE WHEN p_cites THEN p ELSE q END AS citing,
               CASE WHEN p_cites THEN q ELSE p END AS cited,
               collect(DISTINCT toLower(e.name)) AS entities
    """
    with driver.session() as session:
        records = list(session.run(query, ids=list(paper_ids), per_paper=per_paper))
    pairs = {}
    for record in records:
        citing, cited = dict(record["citing"]), dict(record["cited"])
        key = (citing["id"], cited["id"])
        if key not in pairs:
            pairs[key] = (citing, cited, set(record["entities"]))
        else:
            pairs[key][2].update(record["entities"])
    return list(pairs.values())


def rank_citation_candidates(pairs, *, top_k=5, scorer=None, require_subject=True):
    from src.pipeline.near_duplicates import are_near_duplicates

    sentence_pairs, meta = [], []
    for citing, cited, entities in pairs:
        if are_near_duplicates(citing["id"], cited["id"]):
            continue
        for critical in critical_sentences(citing):
            for target in all_sentences(cited):
                subjects = shared_subjects(critical["text"], target["text"], entities)
                if require_subject and not subjects:
                    continue
                sentence_pairs.append((critical["text"], target["text"]))
                meta.append((citing, cited, critical, target, subjects))
    scores = contradiction_probabilities(sentence_pairs, scorer=scorer)
    best = {}
    for (citing, cited, critical, target, subjects), score in zip(meta, scores):
        key = (citing["id"], cited["id"])
        if key not in best or score > best[key][0]:
            best[key] = (float(score), citing, cited, critical, target, subjects)
    ranked = sorted(best.values(), key=lambda item: (-item[0], item[1]["id"], item[2]["id"]))[:top_k]
    return [{
        "paper1": {k: citing.get(k) for k in ("id", "title", "abstract", "year")},
        "paper2": {k: cited.get(k) for k in ("id", "title", "abstract", "year")},
        "evidence_1": critical, "evidence_2": target, "shared_subjects": subjects,
        "nli_contradiction": round(score, 4), "candidate_score": round(score, 4),
        "candidate_method": "citation_critical_nli",
        "flag": f"Citing paper's critical finding vs cited paper (NLI {score:.2f}); shared: {', '.join(subjects[:3])}",
    } for score, citing, cited, critical, target, subjects in ranked]


def detect_contradictions_citation(driver, query, top_k=5, *, retrieve_k=20, **options):
    from src.pipeline.retrieval import nesy_retrieve

    papers = nesy_retrieve(driver, query, top_k=retrieve_k)
    return rank_citation_candidates(citation_neighbours(driver, [p["id"] for p in papers]), top_k=top_k, **options)


def build_corpus_index(driver, corpus, citations, *, scorer=None, batch=500):
    """Corpus-wide index: every critical finding paired with the corpus papers its paper cites.

    ``citations`` maps a citing paper id to the set of cited corpus paper ids (original
    CITES plus fetched reference lists). Subjects are the non-generic SciERC entities
    of either paper that appear in both sentences.
    """
    byid = {p["id"]: {"id": p["id"], "title": p.get("clean_title", p.get("title", "")), "year": p.get("year"),
                      "abstract": p.get("clean_abstract", p.get("abstract", ""))} for p in corpus}
    critical = {pid: critical_sentences(byid[pid]) for pid in citations if pid in byid}
    critical = {pid: rows for pid, rows in critical.items() if rows}
    needed = sorted(set(critical) | {c for pid in critical for c in citations[pid] if c in byid})
    entities = {}
    with driver.session() as session:
        for start in range(0, len(needed), batch):
            for record in session.run(
                    "UNWIND $ids AS pid MATCH (p:Paper {id: pid})-[:MENTIONS]->(e:SciEntity) "
                    "WHERE NOT coalesce(e.generic, false) RETURN pid, collect(toLower(e.name)) AS names",
                    ids=needed[start:start + batch]):
                entities[record["pid"]] = set(record["names"])
    pairs = [(byid[pid], byid[cited], entities.get(pid, set()) | entities.get(cited, set()))
             for pid in critical for cited in sorted(citations[pid]) if cited in byid and cited != pid]
    return rank_citation_candidates(pairs, top_k=len(pairs) or 1, scorer=scorer), {
        "critical_papers": len(critical), "citation_pairs": len(pairs)}


def load_index(path=None):
    import json
    from pathlib import Path

    from src.utils.config import CONTRADICTION_INDEX_FILE

    file = Path(path or CONTRADICTION_INDEX_FILE)
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_candidates_for(index, retrieved_ids, top_k=5):
    """Indexed pairs touching the retrieved papers; prior verdict CONTRADICTION first, then by NLI."""
    retrieved = set(retrieved_ids)
    hits = [row for row in index if row["citing_id"] in retrieved or row["cited_id"] in retrieved]
    hits.sort(key=lambda row: (row.get("verdict") != "CONTRADICTION", -row["nli_contradiction"], row["rank"]))
    return hits[:top_k]


def detect_contradictions_from_index(driver, query, top_k=5, *, retrieve_k=None, index=None):
    """Application path for CONTRADICTION_CANDIDATE_MODE=citation."""
    from src.pipeline.retrieval import nesy_retrieve
    from src.utils.config import CONTRADICTION_INDEX_RETRIEVE_K

    papers = nesy_retrieve(driver, query, top_k=retrieve_k or CONTRADICTION_INDEX_RETRIEVE_K)
    rows = index_candidates_for(index if index is not None else load_index(), [p["id"] for p in papers], top_k)
    if not rows:
        return []
    ids = sorted({row["citing_id"] for row in rows} | {row["cited_id"] for row in rows})
    with driver.session() as session:
        meta = {r["id"]: dict(r) for r in session.run(
            "UNWIND $ids AS pid MATCH (p:Paper {id: pid}) "
            "RETURN p.id AS id, p.title AS title, p.abstract AS abstract, p.year AS year", ids=ids)}
    return [{
        "paper1": meta[row["citing_id"]], "paper2": meta[row["cited_id"]],
        "evidence_1": row["evidence_1"], "evidence_2": row["evidence_2"],
        "nli_contradiction": row["nli_contradiction"], "candidate_score": row["nli_contradiction"],
        "candidate_method": "citation_index", "shared_concept_names": row.get("shared_subjects", []),
        "index_rank": row["rank"], "index_verdict": row.get("verdict"),
    } for row in rows if row["citing_id"] in meta and row["cited_id"] in meta]
