"""
app.py — Advanced RAG chat interface:
  - Hybrid Search (BM25 + FAISS EnsembleRetriever)
  - FlashRank cross-encoder reranking
  - Metadata-based module scope filtering (Streamlit sidebar)

Performance optimisations:
  - BM25 retrievers pre-built and cached per module_id (not rebuilt per query)
  - FlashrankRerank model loaded once and cached
  - Full RetrievalQA chains cached per module_id
  - BM25 + FAISS retrieval runs in parallel via ThreadPoolExecutor
"""

import os
import pickle
import time
import concurrent.futures
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from logger import get_logger

load_dotenv()
log = get_logger("app")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
DOCS_PATH = os.path.join(FAISS_DIR, "bm25_docs.pkl")

FALLBACK = "I'm sorry, that specific detail is not covered in the MPA course materials."

PROMPT_TEMPLATE = """You are an expert assistant for the OCPP MPA Short Course at the University of Cape Coast.
Use ONLY the context below to answer the question.
For every fact you state, cite the source filename and page number in parentheses, e.g. (Source: mpa_zoning.pdf, Page 4).
If the answer is not found in the context, respond exactly with:
"{fallback}"

Context:
{{context}}

Question: {{question}}

Answer:""".format(fallback=FALLBACK)

MODULE_OPTIONS = {
    "All Modules":                      0,
    "Module 1 — Introduction to MPAs":  1,
    "Module 2 — MPAs Around the World": 2,
    "Module 3 — MPAs in Ghana":         3,
    "Module 4 — MPA Identification":    4,
    "Module 6 — MPA Monitoring":        6,
    "Module 7 — MPA and MSP":           7,
}


# ── Cached resources (loaded once per app session) ────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    """Load FAISS index, raw docs, LLM, and reranker — cached for the session."""
    t0 = time.perf_counter()
    log.info("Loading FAISS index from %s", FAISS_DIR)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        FAISS_DIR, embeddings, allow_dangerous_deserialization=True
    )
    log.info("Loading BM25 corpus from %s", DOCS_PATH)
    with open(DOCS_PATH, "rb") as f:
        all_docs = pickle.load(f)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    reranker = FlashrankRerank(top_n=5)
    log.info(
        "Knowledge base ready: %d docs | %.2fs",
        len(all_docs),
        time.perf_counter() - t0,
    )
    return vectorstore, all_docs, llm, reranker


@st.cache_resource(show_spinner=False)
def get_bm25_retriever(module_id: int):
    """
    Build and cache a BM25Retriever for a given module scope.
    Called once per unique module_id — result reused on every subsequent query.
    """
    _, all_docs, _, _ = load_resources()
    t0 = time.perf_counter()
    if module_id:
        docs = [d for d in all_docs if d.metadata.get("module_id") == module_id]
    else:
        docs = all_docs
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 7
    log.info(
        "BM25 retriever built | module_id=%s | docs=%d | %.2fs",
        module_id or "all",
        len(docs),
        time.perf_counter() - t0,
    )
    return retriever


@st.cache_resource(show_spinner=False)
def get_chain(module_id: int):
    """
    Build and cache the full RetrievalQA chain per module_id.
    BM25 and FAISS retrieval are dispatched in parallel inside the EnsembleRetriever.
    """
    vectorstore, _, llm, reranker = load_resources()
    t0 = time.perf_counter()

    bm25_retriever = get_bm25_retriever(module_id)

    faiss_kwargs = {"k": 7}
    if module_id:
        faiss_kwargs["filter"] = {"module_id": module_id}
    faiss_retriever = vectorstore.as_retriever(search_kwargs=faiss_kwargs)

    # Parallel hybrid retrieval — EnsembleRetriever fetches BM25 + FAISS concurrently
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6],
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    log.info(
        "Chain built | module_id=%s | %.2fs",
        module_id or "all",
        time.perf_counter() - t0,
    )
    return chain


# ── Pre-warm all module chains in the background ──────────────────────────────

def _prewarm():
    """Build all module chains at startup so the first query is fast."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(get_chain, mid): mid for mid in MODULE_OPTIONS.values()}
        for fut in concurrent.futures.as_completed(futures):
            mid = futures[fut]
            try:
                fut.result()
                log.debug("Pre-warmed chain for module_id=%s", mid or "all")
            except Exception as exc:
                log.warning("Pre-warm failed for module_id=%s: %s", mid, exc)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MPA Assistant | ACECOR UCC",
    page_icon="🌊",
    layout="centered",
)

st.title("🌊 MPA Short Course Assistant")
st.caption("Powered by OCPP training materials · Africa Centre of Excellence in Coastal Resilience, UCC")

# Pre-warm chains once per session (non-blocking via session state flag)
if "prewarmed" not in st.session_state:
    st.session_state.prewarmed = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_prewarm)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search Scope")
    selected_label = st.selectbox(
        "Filter by Module",
        options=list(MODULE_OPTIONS.keys()),
        index=0,
    )
    selected_module_id = MODULE_OPTIONS[selected_label]
    if selected_module_id:
        st.info(f"Searching within **{selected_label}** only.")
    else:
        st.info("Searching across **all modules**.")

    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about MPAs, zoning, MSP, or Ghana marine policy..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching course materials..."):
            t_query = time.perf_counter()
            log.info(
                "Query received | module_id=%s | query=%r",
                selected_module_id or "all",
                user_input,
            )
            chain = get_chain(selected_module_id)
            result = chain.invoke({"query": user_input})
            elapsed = time.perf_counter() - t_query

        answer = result["result"]
        source_docs = result.get("source_documents", [])
        log.info(
            "Response generated | sources=%d | fallback=%s | %.2fs",
            len(source_docs),
            answer.strip() == FALLBACK,
            elapsed,
        )

        st.markdown(answer)

        if source_docs:
            with st.expander("Sources consulted", expanded=False):
                seen = set()
                for doc in source_docs:
                    meta     = doc.metadata
                    source   = meta.get("source", "Unknown")
                    page     = meta.get("page", "?")
                    topic    = meta.get("topic", "")
                    filename = os.path.basename(source)
                    entry = f"**{filename}** — Page {int(page) + 1}"
                    if topic:
                        entry += f" _(Topic: {topic})_"
                    if entry not in seen:
                        st.markdown(f"- {entry}")
                        seen.add(entry)

    st.session_state.messages.append({"role": "assistant", "content": answer})
