# PROJECT BRIEF: MPA Short Course Intelligence RAG 

## 1. OBJECTIVE
Develop a Retrieval-Augmented Generation (RAG) system that allows users to query a knowledge base of Marine Protected Area (MPA) management materials using Natural Language.

## 2. DATA SOURCE
- **Context:** Training materials from the OCPP MPA Short Course (University of Cape Coast).
- **Format:** PDF documents.
- **Key Themes:** Zoning, Regulation, Marine Spatial Planning (MSP), and Ghana-specific marine policy.

## 3. TECHNICAL STACK
- **Language:** Python 3.10+
- **Orchestration:** LangChain (`langchain`, `langchain-community`, `langchain-openai`)
- **Vector Store:** FAISS (`faiss-cpu`) — local, no C++ build tools required
- **Embeddings:** OpenAI `text-embedding-3-small`
- **LLM:** OpenAI `gpt-4o`
- **Frontend:** Streamlit
- **PDF Loading:** `pypdf` via `langchain-community`
- **Environment:** `python-dotenv`

## 4. FUNCTIONAL REQUIREMENTS
### A. Data Ingestion (`ingest.py`)
1. Load all PDFs from the `/data` folder.
2. Split text into chunks (Size: 1000, Overlap: 100).
3. Generate embeddings and save FAISS index to `./faiss_index`.

### B. Chat Interface (`app.py`)
1. Create a Streamlit chat UI.
2. Load the FAISS index and implement a `RetrievalQA` chain with "stuff" chain type.
3. **CRITICAL:** The system must cite the source filename and page number for every answer.
4. If the information is not in the context, respond: "I'm sorry, that specific detail is not covered in the MPA course materials."

## 5. IMPLEMENTATION NOTES
- All file paths are resolved relative to `__file__` to ensure portability.
- Packages must be installed into the active conda environment using `python -m pip install`, not bare `pip`, to avoid Python interpreter conflicts on Windows.
- ChromaDB was dropped due to `chroma-hnswlib` compilation failures on Windows without MSVC build tools. FAISS provides equivalent functionality with pre-built wheels.