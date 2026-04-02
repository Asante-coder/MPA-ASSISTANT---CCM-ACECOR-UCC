"""
app.py — Streamlit chat interface for the MPA Short Course RAG assistant.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

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


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_qa_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

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
    return chain


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MPA Assistant | ACECOR UCC",
    page_icon="🌊",
    layout="centered",
)

st.title("🌊 MPA Short Course Assistant")
st.caption("Powered by OCPP training materials · Africa Centre of Excellence in Coastal Resilience, UCC")

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
            chain = load_qa_chain()
            result = chain.invoke({"query": user_input})

        answer = result["result"]
        source_docs = result.get("source_documents", [])

        st.markdown(answer)

        # Show source details in an expander
        if source_docs:
            with st.expander("Sources consulted", expanded=False):
                seen = set()
                for doc in source_docs:
                    meta = doc.metadata
                    source = meta.get("source", "Unknown")
                    page = meta.get("page", "?")
                    filename = os.path.basename(source)
                    entry = f"**{filename}** — Page {int(page) + 1}"
                    if entry not in seen:
                        st.markdown(f"- {entry}")
                        seen.add(entry)

    st.session_state.messages.append({"role": "assistant", "content": answer})
