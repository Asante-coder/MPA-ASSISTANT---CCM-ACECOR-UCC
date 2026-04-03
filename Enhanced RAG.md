# 📄 Enhancing a RAG Pipeline to Handle Text Inside Images

## 🎯 Objective
Improve an existing Retrieval-Augmented Generation (RAG) system to:
- Extract text from images embedded in documents (e.g., PDFs, scanned files)
- Integrate extracted text into the semantic search pipeline
- Maintain context and improve retrieval accuracy

---

## ⚠️ Problem Statement
Current pipeline:


### Issue:
- Text embedded in images is not extracted
- Leads to incomplete knowledge base and poor retrieval

---

## ✅ Target Architecture


---

## 🛠️ Implementation Steps

### 1. Extract Text and Images from Documents

#### Recommended Libraries:
- `PyMuPDF (fitz)`
- `pdfplumber`

#### Example:
```python
import fitz  # PyMuPDF

doc = fitz.open("file.pdf")

for page in doc:
    text = page.get_text()
    images = page.get_images(full=True)