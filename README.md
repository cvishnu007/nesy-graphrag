# NeSy-GraphRAG — Automating Review and Hypothesis Generation

PES University Capstone | UE23CS320A/B

---

## Setup

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd nesy-graphrag

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install packages
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Fill in credentials
cp .env .env.local              # edit .env with your real keys
```

---

## Run Order — Do This Exactly Once (Data Pipeline)

```bash
# Step 1 — Fetch 10,000 ArXiv papers → data/arxiv_clean.json
python -m src.ingestion.arxiv_fetcher

# Step 2 — spaCy NER → data/arxiv_ner.json
python -m src.ingestion.ner_extractor

# Step 3 — Embed + store in ChromaDB → data/chromadb/
python -m src.storage.chroma_store

# Step 4 — Insert into Neo4j + create CITES edges
python -m src.storage.neo4j_store
```

## Test the Pipeline

```bash
python -m src.pipeline.orchestrator
```

## Run the App

```bash
streamlit run app/streamlit_app.py
```

---

## Team

- Chiyedu Vishnu (PES1UG23CS169)
- Chinmay Dhar Dwivedi (PES1UG23CS165)
- Dareddy Devesh Reddy (PES1UG23CS171)
- Gurleen Kaur (PES1UG23CS224)

Guide: Prof. Dinesh Singh
