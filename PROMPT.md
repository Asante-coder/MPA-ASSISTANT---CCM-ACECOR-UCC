# PROJECT BRIEF: MPA Short Course Intelligence RAG 

## 1. OBJECTIVE
Develop a Retrieval-Augmented Generation (RAG) system that allows users to query a knowledge base of Marine Protected Area (MPA) management materials using Natural Language.

## 2. DATA SOURCE
- **Context:** Training materials from the OCPP MPA Short Course (University of Cape Coast).
- **Format:** PDF documents.
- **Key Themes:** Zoning, Regulation, Marine Spatial Planning (MSP), and Ghana-specific marine policy.

## 3. TECHNICAL STACK
- **Language:** Python 3.10+
- **Orchestration:** LangChain
- **Vector Store:** ChromaDB (Persistent local storage)
- **Embeddings:** OpenAI `text-embedding-3-small`
- **LLM:** OpenAI `gpt-4o`
- **Frontend:** Streamlit

## 4. FUNCTIONAL REQUIREMENTS
### A. Data Ingestion (`ingest.py`)
1. Load all PDFs from the `/data` folder.
2. Split text into chunks (Size: 1000, Overlap: 100).
3. Generate embeddings and store in `./chroma_db`.

### B. Chat Interface (`app.py`)
1. Create a Streamlit chat UI.
2. Implement a `RetrievalQA` chain with "stuff" chain type.
3. **CRITICAL:** The system must cite the source filename and page number for every answer.
4. If the information is not in the context, respond: "I'm sorry, that specific detail is not covered in the MPA course materials."

## 5. USER PROMPT
@assistant: Please initialize the project by providing the `requirements.txt`. Then, write the full code for `ingest.py` and `app.py` based on these specifications.