# Advanced RAG Optimization Roadmap: MPA Policy Assistant

## 1. Objective
To transition the current "Naive RAG" framework into an "Advanced RAG" system, ensuring high precision in retrieving specific marine governance regulations from the CCM MPA Short Course materials.

## 2. Proposed Strategies

### A. Hybrid Search Implementation
- **Problem:** Current vector search may overlook specific technical terms or Module IDs (e.g., "Module 7", "Zoning").
- **Solution:** Implement a `BM25Retriever` alongside the `Chroma` vector store.
- **Goal:** Combine keyword matching with semantic meaning to ensure 100% accuracy on specific terminology.

### B. Semantic Chunking
- **Problem:** Fixed-size chunking (e.g., 1000 characters) often splits a single regulation across two chunks, losing context.
- **Solution:** Replace `RecursiveCharacterTextSplitter` with `SemanticChunker`. 
- **Goal:** Break text only when the topic changes, keeping related maritime laws in a single context block.

### C. Reranking (Post-Retrieval)
- **Problem:** The top-k results from the vector database are "similar" but not always "most relevant" to the specific query.
- **Solution:** Integrate a Cross-Encoder (e.g., FlashRank or Cohere Rerank) to re-score the top 10 retrieved documents.
- **Goal:** Ensure the LLM only receives the most legally/academically sound evidence.

### D. Metadata Filtering
- **Problem:** As more documents are added, the AI may mix up "Global Trends" (Module 2) with "Ghanaian Regulations" (Module 3).
- **Solution:** Assign metadata tags during `ingest.py`:
  - `source`: "CCM_UCC_MPA_Course"
  - `module_id`: [1-7]
  - `topic`: "Zoning", "MSP", "Monitoring", etc.
- **Goal:** Allow the user to "Scope" their search to a specific module via the Streamlit Sidebar.

## 3. SDLC Implementation Schedule

| Phase | Task | Tools | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Implement Multi-Query Retrieval | LangChain | 🟦 Planned |
| **Phase 2** | Integrate Hybrid Search | BM25 + Chroma | 🟦 Planned |
| **Phase 3** | Deploy Reranking Layer | Cohere/FlashRank| 🟦 Planned |

## 4. Evaluation Metrics
To measure the success of these optimizations, the following will be tracked:
- **Faithfulness:** Does the answer stay true to the CCM documents?
- **Answer Relevance:** Does the answer directly address the user's maritime query?
- **Context Precision:** Are the retrieved snippets actually the ones containing the answer?