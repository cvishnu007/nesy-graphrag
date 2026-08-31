import json
import os
import re
import sys
import time
from typing import Any, Dict, List
from typing import Optional

import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    CLEAN_FILE,
    MAX_AUTHORS,
    MIN_ABSTRACT_WORDS,
    RAW_FILE,
    S2_BATCH_SIZE,
    S2_CHECKPOINT_FILE,
    S2_FIELDS_OF_STUDY,
    S2_INCLUDE_EXISTING,
    S2_LIMIT,
    S2_LIMIT_PER_QUERY,
    S2_MAX_REFS_PER_PAPER,
    S2_PAGE_SIZE,
    S2_PUBLICATION_TYPES,
    S2_QUERY,
    S2_QUERIES,
    S2_SORT,
    S2_YEAR,
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_BASE_URL,
    SEMANTIC_SCHOLAR_MAX_RETRIES,
    SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC,
    SEMANTIC_SCHOLAR_TIMEOUT_SEC,
    is_configured,
)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", "", text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,;:!?()\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


class SemanticScholarClient:
    def __init__(self) -> None:
        self.base_url = SEMANTIC_SCHOLAR_BASE_URL.rstrip("/")
        self.min_interval = max(SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC, 1.0)
        self.timeout_sec = max(SEMANTIC_SCHOLAR_TIMEOUT_SEC, 5)
        self.max_retries = max(SEMANTIC_SCHOLAR_MAX_RETRIES, 1)
        self.last_request_ts = 0.0

        self.session = requests.Session()
        headers = {"User-Agent": "nesy-graphrag/1.0"}
        if is_configured(SEMANTIC_SCHOLAR_API_KEY):
            headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
        self.session.headers.update(headers)

    def _enforce_rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_request_ts
        wait_for = self.min_interval - elapsed
        if wait_for > 0:
            time.sleep(wait_for)

    @staticmethod
    def _parse_retry_after(headers: Dict[str, str], fallback: float = 2.0) -> float:
        value = headers.get("Retry-After", "").strip()
        if not value:
            return fallback
        try:
            return max(float(value), fallback)
        except ValueError:
            return fallback

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            self._enforce_rate_limit()

            try:
                response = self.session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json_body,
                    timeout=self.timeout_sec,
                )
                self.last_request_ts = time.monotonic()
            except requests.RequestException as exc:
                last_error = exc
                sleep_for = min(30.0, 2.0 ** attempt)
                print(f"[S2] Network error ({exc}). retrying in {sleep_for:.1f}s...")
                time.sleep(sleep_for)
                continue

            if response.status_code == 429:
                sleep_for = self._parse_retry_after(response.headers)
                print(f"[S2] 429 rate-limited. sleeping {sleep_for:.1f}s...")
                time.sleep(sleep_for)
                continue

            if response.status_code >= 500:
                sleep_for = min(30.0, 2.0 ** attempt)
                print(f"[S2] Server error {response.status_code}. retrying in {sleep_for:.1f}s...")
                time.sleep(sleep_for)
                continue

            if response.status_code >= 400:
                preview = response.text[:400].replace("\n", " ")
                raise RuntimeError(f"S2 API error {response.status_code}: {preview}")

            return response.json()

        if last_error is not None:
            raise RuntimeError(f"S2 request failed after retries: {last_error}") from last_error
        raise RuntimeError("S2 request failed after retries due to repeated non-success responses.")


def _dedupe_keep_order(items: List[str], max_items: Optional[int] = None) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if max_items is not None and len(out) >= max_items:
            break
    return out


def fetch_seed_papers(
    client: SemanticScholarClient,
    query: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    query = query or S2_QUERY
    limit = S2_LIMIT if limit is None else max(1, int(limit))
    page_size = min(max(S2_PAGE_SIZE, 1), 1000)
    fields = ",".join([
        "paperId",
        "corpusId",
        "externalIds",
        "title",
        "abstract",
        "authors",
        "year",
        "publicationDate",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
        "publicationTypes",
        "citationCount",
        "referenceCount",
        "venue",
        "openAccessPdf",
    ])

    params: Dict[str, Any] = {
        "query": query,
        "limit": page_size,
        "fields": fields,
    }
    if S2_SORT:
        params["sort"] = S2_SORT
    if S2_YEAR:
        params["year"] = S2_YEAR
    if S2_FIELDS_OF_STUDY:
        params["fieldsOfStudy"] = S2_FIELDS_OF_STUDY
    if S2_PUBLICATION_TYPES:
        params["publicationTypes"] = S2_PUBLICATION_TYPES

    papers: List[Dict[str, Any]] = []
    seen_ids = set()
    token = None

    print(f"[S2] Query '{query}': fetching up to {limit} papers...")
    while len(papers) < limit:
        call_params = dict(params)
        if token:
            call_params["token"] = token

        payload = client.request("GET", "/paper/search/bulk", params=call_params)
        batch = payload.get("data") or []

        if not batch:
            break

        for paper in batch:
            paper_id = paper.get("paperId")
            if not paper_id or paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            papers.append(paper)
            if len(papers) >= limit:
                break

        token = payload.get("token")
        print(f"  [S2] {query}: {len(papers)}/{limit} seed papers")
        if not token:
            break

    print(f"[S2] Query '{query}' complete: {len(papers)} papers")
    return papers


def fetch_references_for_papers(
    client: SemanticScholarClient,
    paper_ids: List[str]
) -> Dict[str, List[str]]:
    if not paper_ids:
        return {}

    batch_size = min(max(S2_BATCH_SIZE, 1), 500)
    params = {"fields": "paperId,references.paperId"}
    reference_map: Dict[str, List[str]] = {}

    print("[S2] Enriching papers with references via /paper/batch...")
    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i:i + batch_size]
        payload = {"ids": batch}
        papers = client.request("POST", "/paper/batch", params=params, json_body=payload)

        if isinstance(papers, dict):
            # Defensive fallback if API returns an error payload in JSON.
            raise RuntimeError(f"Unexpected /paper/batch payload: {json.dumps(papers)[:400]}")

        for paper in papers:
            if not paper:
                continue
            pid = paper.get("paperId")
            if not pid:
                continue

            refs = []
            for ref in paper.get("references") or []:
                rid = (ref or {}).get("paperId")
                if rid and rid != pid:
                    refs.append(rid)

            reference_map[pid] = _dedupe_keep_order(refs, max_items=S2_MAX_REFS_PER_PAPER)

        print(f"  [S2] Reference batches: {min(i + batch_size, len(paper_ids))}/{len(paper_ids)}")

    return reference_map


def _build_categories(paper: Dict[str, Any]) -> List[str]:
    categories = []
    for c in paper.get("fieldsOfStudy") or []:
        if isinstance(c, str) and c.strip():
            categories.append(c.strip())
    for c in paper.get("s2FieldsOfStudy") or []:
        if isinstance(c, dict):
            label = (c.get("category") or "").strip()
            if label:
                categories.append(label)
    return _dedupe_keep_order(categories)


def normalize_papers(
    raw_papers: List[Dict[str, Any]],
    reference_map: Dict[str, List[str]],
    ingestion_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    normalized = []

    for paper in raw_papers:
        paper_id = paper.get("paperId")
        if not paper_id:
            continue

        authors = paper.get("authors") or []
        author_names = []
        author_ids = []
        for author in authors:
            if not isinstance(author, dict):
                continue
            name = (author.get("name") or "").strip()
            aid = (author.get("authorId") or "").strip()
            if name:
                author_names.append(name)
            if aid:
                author_ids.append(aid)

        external_ids = paper.get("externalIds") or {}
        doi = external_ids.get("DOI") if isinstance(external_ids, dict) else None
        pub_date = paper.get("publicationDate")
        year = paper.get("year")

        normalized.append({
            "id": paper_id,
            "paperId": paper_id,
            "corpusId": paper.get("corpusId"),
            "title": (paper.get("title") or "").strip(),
            "abstract": (paper.get("abstract") or "").strip(),
            "authors": _dedupe_keep_order(author_names, max_items=MAX_AUTHORS),
            "author_ids": _dedupe_keep_order(author_ids, max_items=MAX_AUTHORS),
            "categories": _build_categories(paper),
            "doi": doi,
            "published": pub_date if pub_date else (f"{year}-01-01" if year else None),
            "pdf_url": ((paper.get("openAccessPdf") or {}).get("url") if isinstance(paper.get("openAccessPdf"), dict) else None),
            "venue": paper.get("venue"),
            "publicationTypes": paper.get("publicationTypes") or [],
            "citationCount": int(paper.get("citationCount") or 0),
            "referenceCount": int(paper.get("referenceCount") or 0),
            "references": reference_map.get(paper_id, []),
            "ingestion_queries": [ingestion_query] if ingestion_query else [],
            "source": "semantic_scholar",
        })

    return normalized


def _merge_unique(left: Any, right: Any) -> List[Any]:
    left_items = left if isinstance(left, list) else []
    right_items = right if isinstance(right, list) else []
    return list(dict.fromkeys(item for item in left_items + right_items if item))


def merge_paper_records(
    existing: Dict[str, Dict[str, Any]],
    incoming: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Merge normalized papers by S2 paper ID without losing topic provenance."""
    for paper in incoming:
        paper_id = str(paper.get("id") or paper.get("paperId") or "").strip()
        if not paper_id:
            continue

        current = existing.get(paper_id)
        if current is None:
            item = dict(paper)
            item["ingestion_queries"] = _merge_unique(
                [], item.get("ingestion_queries")
            )
            existing[paper_id] = item
            continue

        current["references"] = _merge_unique(
            current.get("references"), paper.get("references")
        )
        current["categories"] = _merge_unique(
            current.get("categories"), paper.get("categories")
        )
        current["ingestion_queries"] = _merge_unique(
            current.get("ingestion_queries"), paper.get("ingestion_queries")
        )
        current["citationCount"] = max(
            int(current.get("citationCount") or 0),
            int(paper.get("citationCount") or 0),
        )
        current["referenceCount"] = max(
            int(current.get("referenceCount") or 0),
            int(paper.get("referenceCount") or 0),
        )
        for field in ("abstract", "title", "doi", "venue", "pdf_url"):
            if not current.get(field) and paper.get(field):
                current[field] = paper[field]

    return existing


def _atomic_json_write(path: str, value: Any) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
    os.replace(temporary, path)


def _load_existing_records() -> Dict[str, Dict[str, Any]]:
    if not S2_INCLUDE_EXISTING or not os.path.exists(RAW_FILE):
        return {}
    with open(RAW_FILE, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise RuntimeError(f"{RAW_FILE} must contain a JSON list")

    existing: Dict[str, Dict[str, Any]] = {}
    legacy = []
    for paper in payload:
        if not isinstance(paper, dict):
            continue
        item = dict(paper)
        if not item.get("ingestion_queries"):
            item["ingestion_queries"] = [S2_QUERY]
            legacy.append(item)
        merge_paper_records(existing, [item])
    print(f"[S2] Loaded {len(existing)} existing raw papers for merge")
    if legacy:
        print(f"[S2] Added legacy query provenance to {len(legacy)} records")
    return existing


def _load_checkpoint() -> Dict[str, Any]:
    if not os.path.exists(S2_CHECKPOINT_FILE):
        return {"completed": []}
    with open(S2_CHECKPOINT_FILE, "r", encoding="utf-8") as file:
        checkpoint = json.load(file)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{S2_CHECKPOINT_FILE} must contain a JSON object")
    checkpoint.setdefault("completed", [])
    return checkpoint


def _checkpoint_key(query: str) -> str:
    return f"{query}|limit={S2_LIMIT_PER_QUERY}|year={S2_YEAR}"


def preprocess(papers: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(papers)
    print(f"Before cleaning : {len(df)} papers")

    if df.empty:
        os.makedirs(os.path.dirname(CLEAN_FILE), exist_ok=True)
        df.to_json(CLEAN_FILE, orient="records", indent=2)
        print(f"Saved empty dataset to {CLEAN_FILE}")
        return df

    df.dropna(subset=["title", "abstract"], inplace=True)
    df = df[df["title"].str.strip() != ""]
    df = df[df["abstract"].str.strip() != ""]
    df.drop_duplicates(subset=["id"], inplace=True)

    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_abstract"] = df["abstract"].apply(clean_text)
    df["abstract_word_count"] = df["clean_abstract"].str.split().str.len()
    df = df[df["abstract_word_count"] >= MIN_ABSTRACT_WORDS]

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        df["year"] = None

    missing_year = df["year"].isna()
    if missing_year.any():
        parsed = pd.to_datetime(df.loc[missing_year, "published"], errors="coerce", utc=True)
        df.loc[missing_year, "year"] = parsed.dt.year
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    df["categories"] = df["categories"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df["references"] = df["references"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df["primary_category"] = df["categories"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "unknown"
    )

    print(f"After cleaning  : {len(df)} papers")
    os.makedirs(os.path.dirname(CLEAN_FILE), exist_ok=True)
    df.to_json(CLEAN_FILE, orient="records", indent=2)
    print(f"Saved to {CLEAN_FILE}")
    return df


def run() -> pd.DataFrame:
    print("=== Semantic Scholar ingestion ===")
    if not is_configured(SEMANTIC_SCHOLAR_API_KEY):
        print("[S2] Warning: SEMANTIC_SCHOLAR_API_KEY not set. Unauthenticated calls may be heavily limited.")

    client = SemanticScholarClient()
    merged = _load_existing_records()
    checkpoint = _load_checkpoint()
    completed = set(checkpoint.get("completed") or [])
    if completed and not merged:
        print(
            "[S2] Checkpoint exists without a reusable raw dataset; "
            "starting all topics again."
        )
        completed.clear()

    print(
        f"[S2] Multi-topic plan: {len(S2_QUERIES)} queries, "
        f"up to {S2_LIMIT_PER_QUERY} papers per query"
    )
    for index, query in enumerate(S2_QUERIES, start=1):
        key = _checkpoint_key(query)
        if key in completed:
            print(f"[S2] [{index}/{len(S2_QUERIES)}] Resume: skipping '{query}'")
            continue

        print(f"\n[S2] [{index}/{len(S2_QUERIES)}] Starting '{query}'")
        seed_papers = fetch_seed_papers(
            client,
            query=query,
            limit=S2_LIMIT_PER_QUERY,
        )
        paper_ids = [
            paper.get("paperId")
            for paper in seed_papers
            if paper.get("paperId")
        ]
        reference_map = {
            paper_id: list(merged[paper_id].get("references") or [])
            for paper_id in paper_ids
            if paper_id in merged
        }
        new_paper_ids = [paper_id for paper_id in paper_ids if paper_id not in merged]
        reference_map.update(
            fetch_references_for_papers(client, new_paper_ids)
        )
        normalized = normalize_papers(
            seed_papers,
            reference_map,
            ingestion_query=query,
        )
        before = len(merged)
        merge_paper_records(merged, normalized)
        added = len(merged) - before

        _atomic_json_write(RAW_FILE, list(merged.values()))
        completed.add(key)
        checkpoint = {
            "completed": sorted(completed),
            "queries": list(S2_QUERIES),
            "limit_per_query": S2_LIMIT_PER_QUERY,
            "year": S2_YEAR,
            "unique_papers": len(merged),
        }
        _atomic_json_write(S2_CHECKPOINT_FILE, checkpoint)
        print(
            f"[S2] Saved checkpoint: {len(normalized)} fetched, "
            f"{added} new, {len(merged)} unique total"
        )

    normalized = list(merged.values())
    _atomic_json_write(RAW_FILE, normalized)
    print(
        f"[S2] Saved merged raw dataset ({len(normalized)} unique records) "
        f"to {RAW_FILE}"
    )

    df = preprocess(normalized)

    if not df.empty:
        print("\n=== Sanity Check ===")
        print("Shape          :", df.shape)
        print("\nTop categories :\n", df["primary_category"].value_counts().head(10))
        print("\nPapers per year:\n", df["year"].value_counts().sort_index())
        print("\nSample title   :", df["title"].iloc[0][:120])

    return df


if __name__ == "__main__":
    run()
