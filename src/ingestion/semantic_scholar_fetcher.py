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
    S2_FIELDS_OF_STUDY,
    S2_LIMIT,
    S2_MAX_REFS_PER_PAPER,
    S2_PAGE_SIZE,
    S2_PUBLICATION_TYPES,
    S2_QUERY,
    S2_SORT,
    S2_YEAR,
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_BASE_URL,
    SEMANTIC_SCHOLAR_MAX_RETRIES,
    SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC,
    SEMANTIC_SCHOLAR_TIMEOUT_SEC,
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
        if SEMANTIC_SCHOLAR_API_KEY:
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


def fetch_seed_papers(client: SemanticScholarClient) -> List[Dict[str, Any]]:
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
        "query": S2_QUERY,
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

    print(f"[S2] Fetching up to {S2_LIMIT} papers from /paper/search/bulk...")
    while len(papers) < S2_LIMIT:
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
            if len(papers) >= S2_LIMIT:
                break

        token = payload.get("token")
        print(f"  [S2] Seed papers fetched: {len(papers)}")
        if not token:
            break

    print(f"[S2] Seed fetch complete: {len(papers)} papers")
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
    reference_map: Dict[str, List[str]]
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
            "source": "semantic_scholar",
        })

    return normalized


def preprocess(papers: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(papers)
    print(f"Before cleaning : {len(df)} papers")

    if df.empty:
        os.makedirs(os.path.dirname(CLEAN_FILE), exist_ok=True)
        df.to_json(CLEAN_FILE, orient="records", indent=2)
        print(f"Saved empty dataset → {CLEAN_FILE}")
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
    print(f"Saved → {CLEAN_FILE}")
    return df


def run() -> pd.DataFrame:
    print("=== Semantic Scholar ingestion ===")
    if not SEMANTIC_SCHOLAR_API_KEY:
        print("[S2] Warning: SEMANTIC_SCHOLAR_API_KEY not set. Unauthenticated calls may be heavily limited.")

    client = SemanticScholarClient()
    seed_papers = fetch_seed_papers(client)
    paper_ids = [p.get("paperId") for p in seed_papers if p.get("paperId")]
    reference_map = fetch_references_for_papers(client, paper_ids)

    normalized = normalize_papers(seed_papers, reference_map)
    os.makedirs(os.path.dirname(RAW_FILE), exist_ok=True)
    with open(RAW_FILE, "w") as f:
        json.dump(normalized, f, indent=2)
    print(f"[S2] Saved normalized raw dataset ({len(normalized)} records) → {RAW_FILE}")

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
