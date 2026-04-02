"""
ingest.py — Load PDFs from /data, chunk, embed, and persist to a FAISS index.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def load_pdfs(data_dir: str):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded: {filename} ({len(docs)} pages)")
    return documents


def main():
    print("=== MPA Knowledge Base Ingestion ===")

    print("\n[1/3] Loading PDFs...")
    documents = load_pdfs(DATA_DIR)
    print(f"Total pages loaded: {len(documents)}")

    print("\n[2/3] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    print("\n[3/3] Generating embeddings and saving FAISS index...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(FAISS_DIR)
    print(f"FAISS index saved to: {FAISS_DIR}")
    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
