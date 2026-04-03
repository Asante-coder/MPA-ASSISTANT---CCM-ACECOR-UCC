# MPA Short Course Intelligence Assistant (RAG Framework)

## Overview
This project is a **Retrieval-Augmented Generation (RAG)** application designed to assist Marine Protected Area (MPA) managers and researchers. It allows users to interact with specialized marine policy and spatial planning materials using natural language.

The system was developed to bridge the gap between complex coastal governance documentation and real-time decision support, specifically focusing on the **OCPP Marine Protected Areas Short Course** curriculum from the **Centre for Coastal Management (CCM), University of Cape Coast, Ghana.**

## Technical Stack
| Component | Module |
|---|---|
| LLM + Vision OCR | `gpt-4o` via `langchain-openai` |
| Embeddings | `text-embedding-3-small` via `langchain-openai` |
| Vector Store | `faiss-cpu` (local, no compilation required) |
| PDF + Image Extraction | `pymupdf (fitz)` |
| Semantic Chunking | `SemanticChunker` via `langchain-experimental` |
| Keyword Search | `BM25Retriever` via `rank_bm25` |
| Hybrid Retrieval | `EnsembleRetriever` via `langchain-classic` |
| Reranking | `FlashrankRerank` via `flashrank` |
| Chains & Prompts | `RetrievalQA`, `PromptTemplate` via `langchain-classic` |
| Orchestration | `langchain` + `langchain-community` |
| Logging | Python `logging` with rotating file handler |
| Frontend | `streamlit` |
| Environment | `python-dotenv` |

## Key Features
- **Vision-Enhanced Ingestion:** Uses GPT-4o Vision to OCR text embedded in images, diagrams, and figures inside PDFs — text that standard PDF loaders miss entirely.
- **Hybrid Search:** Combines BM25 keyword matching (40%) with FAISS semantic search (60%) via `EnsembleRetriever` for both precise term matching and conceptual relevance.
- **Semantic Chunking:** Splits documents at topic boundaries using `SemanticChunker` instead of fixed character counts, keeping related content together.
- **Cross-Encoder Reranking:** FlashRank reranks the top 14 retrieved chunks to the best 5 before passing to the LLM, improving answer precision.
- **Module Scope Filtering:** Sidebar lets users scope queries to a specific module (e.g. "Module 3 — MPAs in Ghana") via metadata filtering on both FAISS and BM25.
- **Performance Optimisation:** BM25 retrievers, the reranker, and full QA chains are cached per module using `@st.cache_resource`. All module chains are pre-warmed at startup via `ThreadPoolExecutor` so the first query is as fast as subsequent ones.
- **Source Attribution:** Every response cites the source filename, page number, and topic.
- **Structured Logging:** All key events (queries, responses, load times) are written to `logs/mpa_assistant.log` with a rotating file handler.
- **Streamlit Cloud Compatible:** Uses `langchain 1.x` + `langchain-classic` which supports Python 3.14 via Pydantic v2 — no pydantic.v1 compatibility issues.

## RAG Architecture
```
PDF Documents
     │
     ▼
PyMuPDF (text + image extraction)
     │
     ├── Regular page text
     └── Embedded images ──► GPT-4o Vision OCR
                                    │
                              Combined page text
                                    │
                            SemanticChunker
                          (topic-aware splits)
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
            FAISS Vector Index              BM25 Corpus (.pkl)
          (text-embedding-3-small)         (keyword index)
                    │                                │
                    └───────────────┬────────────────┘
                                    ▼
                          EnsembleRetriever
                         (40% BM25 + 60% FAISS)
                                    │
                            FlashRank Reranker
                              (top 14 → top 5)
                                    │
                               GPT-4o LLM
                          (stuff chain + prompt)
                                    │
                         Cited Answer + Sources
```

## Project Structure
```text
├── data/                   # Knowledge base (MPA Course PDFs)
├── faiss_index/
│   ├── index.faiss         # FAISS vector index
│   ├── index.pkl           # FAISS metadata
│   └── bm25_docs.pkl       # BM25 document corpus
├── logs/
│   └── mpa_assistant.log   # Rotating application log
├── ingest.py               # PDF + image ingestion, chunking & embedding
├── app.py                  # Streamlit chat interface
├── logger.py               # Shared logging configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup & Usage

### 1. Install dependencies
```bash
conda activate (your python env)
python -m pip install -r requirements.txt
```

> **Windows note:** Always use `python -m pip` to ensure packages install into the active conda environment, not the system Python.

### 2. Set your OpenAI API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

### 3. Build the vector index (run once, or when PDFs change)
```bash
python ingest.py
```
This extracts text and runs GPT-4o Vision on all embedded images, then builds the FAISS index and BM25 corpus. Re-run whenever new PDFs are added to `data/`.

### 4. Launch the app
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment
The app is compatible with Streamlit Community Cloud. Ensure:
- `OPENAI_API_KEY` is set in the app's **Secrets** settings (not in `.env`)
- The `faiss_index/` folder and its three files are committed to the repository (the index is pre-built — ingestion does not run on the cloud)

## Author
Francis Asante Nsiah
