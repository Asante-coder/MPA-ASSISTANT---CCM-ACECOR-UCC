"""
ingest.py — Enhanced RAG ingestion:
  - PyMuPDF extracts both regular text AND images from every PDF page
  - GPT-4o Vision OCRs embedded images (diagrams, figures, scanned text)
  - Extracted image text is merged with page text for a complete knowledge base
  - Semantic chunking (SemanticChunker) — topic-aware splits
  - Rich metadata tagging (source_id, module_id, topic)
  - FAISS vector index + BM25 corpus saved for hybrid search
"""

import os
import re
import pickle
import base64
import concurrent.futures
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from logger import get_logger

load_dotenv()
log = get_logger("ingest")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
DOCS_PATH = os.path.join(FAISS_DIR, "bm25_docs.pkl")

# Minimum image size to attempt OCR — skip tiny icons/logos
MIN_IMG_WIDTH  = 100
MIN_IMG_HEIGHT = 100

# Max parallel Vision API calls
MAX_VISION_WORKERS = 4

MODULE_TOPICS = {
    1: "Introduction to MPAs",
    2: "MPAs Around the World",
    3: "MPAs in Ghana",
    4: "MPA Identification",
    5: "MPA Management",
    6: "MPA Monitoring",
    7: "MPA and MSP",
}

_openai_client = OpenAI()


def extract_module_id(filename: str) -> int:
    match = re.search(r"Module\s+(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def image_to_base64(pix: fitz.Pixmap) -> str:
    """Convert a PyMuPDF Pixmap to a base64-encoded PNG string."""
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ocr_image_with_vision(b64_image: str, context: str) -> str:
    """
    Send a base64 image to GPT-4o Vision and extract any text/data it contains.
    `context` is the surrounding page text, used to ground the model.
    """
    try:
        response = _openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This image is from a Marine Protected Area (MPA) training document. "
                                "Extract ALL text, labels, data, table contents, and key information visible in this image. "
                                "If the image contains a diagram or figure, describe it and list all text labels. "
                                "Return ONLY the extracted text and descriptions — no commentary.\n\n"
                                f"Page context: {context[:300]}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        log.warning("Vision OCR failed for an image: %s", exc)
        return ""


def extract_page_images(page: fitz.Page, page_text: str) -> list[str]:
    """
    Extract all qualifying images from a PDF page and OCR them in parallel.
    Returns a list of extracted text strings (one per image).
    """
    image_list = page.get_images(full=True)
    if not image_list:
        return []

    b64_images = []
    doc = page.parent
    for img_info in image_list:
        xref = img_info[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.width < MIN_IMG_WIDTH or pix.height < MIN_IMG_HEIGHT:
                continue
            if pix.n > 4:  # CMYK or other — convert to RGB first
                pix = fitz.Pixmap(fitz.csRGB, pix)
            b64_images.append(image_to_base64(pix))
        except Exception as exc:
            log.debug("Skipping image xref=%d: %s", xref, exc)

    if not b64_images:
        return []

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_VISION_WORKERS) as pool:
        futures = [pool.submit(ocr_image_with_vision, b64, page_text) for b64 in b64_images]
        for fut in concurrent.futures.as_completed(futures):
            text = fut.result()
            if text:
                results.append(text)
    return results


def load_pdf_with_vision(filepath: str, filename: str) -> list[Document]:
    """
    Load a PDF using PyMuPDF, extracting both regular text and image text per page.
    Returns a list of LangChain Documents (one per page).
    """
    module_id = extract_module_id(filename)
    topic = MODULE_TOPICS.get(module_id, "General")
    base_metadata = {
        "source":    filepath,
        "source_id": "CCM_UCC_MPA_Course",
        "module_id": module_id,
        "topic":     topic,
    }

    documents = []
    pdf = fitz.open(filepath)
    total_images = 0

    for page_num, page in enumerate(pdf):
        page_text = page.get_text("text").strip()
        image_texts = extract_page_images(page, page_text)
        total_images += len(image_texts)

        if image_texts:
            image_block = "\n\n[Image/Figure Content]\n" + "\n\n".join(image_texts)
            combined_text = page_text + image_block
        else:
            combined_text = page_text

        if not combined_text.strip():
            continue

        documents.append(Document(
            page_content=combined_text,
            metadata={**base_metadata, "page": page_num},
        ))

    pdf.close()
    log.info(
        "Loaded: %s | pages=%d | images OCR'd=%d",
        filename, len(documents), total_images,
    )
    return documents


def load_all_pdfs(data_dir: str) -> list[Document]:
    documents = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            docs = load_pdf_with_vision(filepath, filename)
            documents.extend(docs)
    return documents


def main():
    log.info("=== MPA Knowledge Base Ingestion (Enhanced RAG + Vision OCR) ===")

    log.info("[1/3] Loading PDFs with text + image extraction...")
    documents = load_all_pdfs(DATA_DIR)
    log.info("Total pages loaded: %d", len(documents))

    log.info("[2/3] Semantic chunking (topic-aware splits)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    chunks = splitter.split_documents(documents)
    log.info("Total semantic chunks created: %d", len(chunks))

    log.info("[3/3] Building FAISS index and saving BM25 corpus...")
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(FAISS_DIR)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    log.info("FAISS index  -> %s", FAISS_DIR)
    log.info("BM25 corpus  -> %s", DOCS_PATH)
    log.info("Ingestion complete.")


if __name__ == "__main__":
    main()
