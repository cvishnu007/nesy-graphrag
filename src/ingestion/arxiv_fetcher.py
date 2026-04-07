import os
import re
import json
import arxiv
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import RAW_FILE, CLEAN_FILE, MIN_ABSTRACT_WORDS

# ── All 40 CS categories from your notebook ───────────
CS_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.IR", "cs.CL", "cs.CV", "cs.DB", "cs.MA",
    "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CR", "cs.CY", "cs.DC",
    "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR",
    "cs.GT", "cs.HC", "cs.IT", "cs.LO", "cs.MM", "cs.MS", "cs.NA",
    "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO",
    "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"
]

# ── 2000 papers per year, 2020–2024 = 10,000 total ────
YEAR_RANGES = [
    ("20200101", "20201231", 2000),
    ("20210101", "20211231", 2000),
    ("20220101", "20221231", 2000),
    ("20230101", "20231231", 2000),
    ("20240101", "20241231", 2000),
]


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,;:!?()\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def fetch_papers():
    cat_query = " OR ".join([f"cat:{c}" for c in CS_CATEGORIES])
    all_papers = []

    for start_date, end_date, max_results in YEAR_RANGES:
        year = start_date[:4]
        print(f"\nFetching {year} papers...")

        full_query = f"({cat_query}) AND submittedDate:[{start_date} TO {end_date}]"

        client = arxiv.Client(page_size=500, delay_seconds=3.0, num_retries=5)
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        year_papers = []
        for i, result in enumerate(client.results(search)):
            year_papers.append({
                "id"         : result.entry_id,
                "title"      : result.title,
                "abstract"   : result.summary,
                "authors"    : [a.name for a in result.authors],
                "categories" : result.categories,
                "doi"        : result.doi,
                "published"  : str(result.published),
                "pdf_url"    : result.pdf_url
            })
            if (i + 1) % 500 == 0:
                print(f"  {year}: Fetched {i + 1} papers...")

        print(f"  {year}: Done — {len(year_papers)} papers")
        all_papers.extend(year_papers)

    os.makedirs(os.path.dirname(RAW_FILE), exist_ok=True)
    with open(RAW_FILE, "w") as f:
        json.dump(all_papers, f, indent=2)

    print(f"\nTotal fetched: {len(all_papers)} papers → {RAW_FILE}")
    return all_papers


def preprocess(papers):
    df = pd.DataFrame(papers)
    print(f"Before cleaning : {len(df)} papers")

    df.dropna(subset=["title", "abstract"], inplace=True)
    df = df[df["abstract"].str.strip() != ""]
    df.drop_duplicates(subset=["id"], inplace=True)

    df["clean_title"]         = df["title"].apply(clean_text)
    df["clean_abstract"]      = df["abstract"].apply(clean_text)
    df["abstract_word_count"] = df["clean_abstract"].str.split().str.len()
    df = df[df["abstract_word_count"] >= MIN_ABSTRACT_WORDS]

    df["year"] = pd.to_datetime(df["published"], utc=True).dt.year
    df["primary_category"] = df["categories"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "unknown"
    )

    print(f"After cleaning  : {len(df)} papers")
    df.to_json(CLEAN_FILE, orient="records", indent=2)
    print(f"Saved → {CLEAN_FILE}")
    return df


if __name__ == "__main__":
    print("=== Step 1: Fetching papers ===")
    papers = fetch_papers()

    print("\n=== Step 2: Preprocessing ===")
    df = preprocess(papers)

    print("\n=== Sanity Check ===")
    print("Shape          :", df.shape)
    print("\nTop categories :\n", df["primary_category"].value_counts().head(10))
    print("\nPapers per year:\n", df["year"].value_counts().sort_index())
    print("\nSample abstract:\n", df["clean_abstract"].iloc[0])
