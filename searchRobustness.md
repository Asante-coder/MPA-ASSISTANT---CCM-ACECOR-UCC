# 🧠 Prompt for Coding Agent: Improve Search Robustness (Case & Whitespace Insensitivity)

## 🎯 Objective
Enhance the existing search/retrieval system (RAG pipeline) to:
- Ignore **word capitalization (case-insensitive search)**
- Ignore **extra/missing whitespace differences**
- Improve matching consistency across OCR text, user queries, and indexed documents

---

## ⚠️ Problem
Current system likely performs **strict string or embedding matching**, leading to failures such as:

- "Climate Change" ≠ "climate change"
- "TotalRevenue" ≠ "Total Revenue"
- "marine   spatial plan" ≠ "marine spatial plan"

This is especially problematic for:
- OCR-extracted text (often inconsistent spacing)
- User queries (unpredictable formatting)

---

## ✅ Required Improvements

### 1. Normalize Text During Ingestion

Apply normalization BEFORE:
- Chunking
- Embedding generation
- Indexing

#### Rules:
- Convert all text to lowercase
- Collapse multiple spaces into one
- Strip leading/trailing whitespace

#### Example:
```python
import re

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    return text.strip()