# MPA Short Course Intelligence Assistant (RAG Framework)

## Overview
This project is a **Retrieval-Augmented Generation (RAG)** application designed to assist Marine Protected Area (MPA) managers and researchers. It allows users to interact with specialized marine policy and spatial planning materials using natural language.

The system was developed to bridge the gap between complex coastal governance documentation and real-time decision support, specifically focusing on the **OCPP Marine Protected Areas Short Course** curriculum from the **Centre for Coastal Management (CCM), University of Cape Coast, Ghana.**

## Key Features
- **Semantic Search:** Uses OpenAI embeddings to find relevant policy details even if exact keywords aren't used.
- **Context-Aware Chat:** Powered by GPT-4o, the assistant only answers based on the provided course modules to ensure accuracy.
- **Source Attribution:** Every response includes citations (Source file and Page number) to maintain academic and legal integrity.
- **Streamlit Interface:** A clean, web-based UI for easy interaction.

## Project Structure
```text
├── data/               # Knowledge base (MPA Course PDFs)
├── chroma_db/          # Persistent Vector Database
├── ingest.py           # Script for PDF processing & embedding
├── app.py              # Streamlit Chat Interface
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

## Author
Francis Asante Nsiah