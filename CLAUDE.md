# CLAUDE.md


This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered resume screening system that ranks candidates against a job description using semantic similarity, 3-layer skill matching, experience analysis, and domain-aware scoring weights.

## Commands

### Run the CLI (batch processing)
```bash
cd src
python batch_test.py                        # uses default resumes/data folder + interactive JD input
python batch_test.py --folder "path/to/pdfs" --jd "path/to/jd.txt"
```

### Run the Streamlit dashboard
```bash
streamlit run src/Dashboard.py
```

### Test a single PDF
```bash
cd src
python test_parser.py
```

### Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### External binaries required (Windows)
- **Tesseract-OCR** installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`
- **Poppler** installed at `C:\Program Files\poppler-25.11.0\Library\bin`

These paths are hardcoded in `src/pdf_parser.py:11-12`.

## Architecture

```
INPUT → PARSE → EXTRACT → MATCH → SCORE → RANK → EXPORT
```

### Pipeline (in order)

1. **`pdf_parser.py`** — Extracts text from PDFs. Primary: `pdfplumber` with column-aware splitting (detects 2-column layouts). Fallback: Tesseract OCR via `pdf2image`. Returns `ParsedDocument(text, method, confidence, page_count)`.

2. **`ner_extractor.py`** — Extracts name (spaCy `en_core_web_sm` NER), email, phone, LinkedIn, GitHub from raw text.

3. **`information_extractor.py`** — Extracts skills list via regex against a built-in skill vocabulary (~50+ skills).

4. **`experience_extractor.py`** — Parses work history dates (`python-dateutil`), computes total years and role-specific months.

5. **`domain_config.py`** — Detects job domain from JD keywords (fast path: 10 built-in configs). Falls back to Anthropic API for unknown domains. The API key placeholder is at line 272 (`"x-api-key": "YOUR_API_KEY_HERE"`).

6. **`skill_matcher.py`** — 3-layer skill matching:
   - Layer 1: Exact match (with alias normalization, e.g. `sklearn → scikit-learn`)
   - Layer 2: Fuzzy match via `rapidfuzz` (threshold: 85)
   - Layer 3: Semantic cosine similarity via `sentence-transformers/all-MiniLM-L6-v2` (threshold: 0.72)
   - Embeddings are disk-cached in `cache/skill_embeddings/` (MD5 keyed pickles).

7. **`matcher.py`** — Combines all scores into a final weighted score:
   - Semantic score (resume header 1500 chars vs full JD, MiniLM)
   - Skill score (from skill_matcher)
   - Total experience score (non-linear, penalizes underqualified, caps overqualified)
   - Role-specific experience score
   - Weights vary by domain (e.g., ML Engineer: 20% semantic + 35% skills + 15% total_exp + 30% role_exp)
   - Decision thresholds: ≥0.78 Highly Recommended, 0.60–0.77 Qualified, 0.42–0.59 Maybe, <0.42 Reject

8. **`ranker.py`** — Batch pipeline: collects PDFs recursively, runs pipeline with error isolation (one bad PDF won't stop the batch), sorts by score, exports JSON + CSV to `outputs/results_YYYYMMDD_HHMMSS.{json,csv}`.

9. **`Dashboard.py`** — Streamlit web UI. Upload PDFs, paste JD, view radar chart + score distribution + skill breakdown. Dark theme with custom CSS.

### Key Design Details

- **Lazy model loading**: `sentence-transformers` model loads once globally (`get_model()`) — expect ~3s cold start.
- **Skill implication graph**: Certain skills imply others (e.g., TensorFlow → Python, PyTorch → deep learning); this is in `skill_matcher.py`.
- **Domain scoring weights**: Defined in `domain_config.py`. To add a new domain, add a new entry to the domain config dict there.
- **Non-linear experience scoring**: Uses `ratio^1.5` for underqualified candidates; capped at 1.0 for overqualified.

## Outputs

- Results export to `outputs/` (gitignored)
- Uploaded resumes go to `temp_uploads/` (gitignored)
- Embedding cache in `cache/` (gitignored)
