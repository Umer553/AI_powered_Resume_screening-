# RecruitIQ — AI-Powered Resume Screening System

An end-to-end resume screening pipeline that ranks candidates against a job description using semantic similarity, multi-layer skill matching, experience analysis, and domain-aware scoring weights — with a BERT-based NER layer for accurate candidate identity extraction.

---

## Architecture

```
PDF → PARSE → NER (BERT + fallback) → SKILLS → EXPERIENCE → SCORE → RANK → EXPORT
```

| Stage | Module | What it does |
|---|---|---|
| Parse | `pdf_parser.py` | pdfplumber (2-column aware) → Tesseract OCR fallback |
| Identity | `ner_extractor.py` | 4-layer name extraction + BERT multi-entity (ORG, LOC) |
| Skills | `information_extractor.py` | Regex against 80+ skill vocabulary |
| Experience | `experience_extractor.py` | Date-range parsing, overlap correction, role mapping |
| Matching | `skill_matcher.py` | 3-layer: exact → fuzzy (rapidfuzz) → semantic (MiniLM) |
| Scoring | `matcher.py` | Domain-aware weighted score (semantic + skill + exp) |
| Ranking | `ranker.py` | Batch pipeline, error isolation, JSON/CSV export |
| Dashboard | `Dashboard.py` | Streamlit UI with radar chart, analytics, NER badges |

---

## Novel AI Layer — BERT NER (dslim/bert-base-NER)

The key research contribution is a **4-layer cascading NER pipeline** for candidate identity extraction:

```
Layer 0  dslim/bert-base-NER  via HuggingFace API   confidence 0.95   ← NEW
Layer 1  spaCy en_core_web_sm                        confidence 0.90
Layer 2  First-line heuristic                        confidence 0.75
Layer 3  Regex capitalized pattern                   confidence 0.60
```

**Why BERT over spaCy alone:**
- spaCy `en_core_web_sm` miss rate ~40% on non-Western names (South Asian, Arabic, East Asian)
- BERT reads a full 512-token context window — handles name anywhere in the document
- Sliding window scan (4 overlapping windows) catches names buried past the header
- Adjacent PER token merging: `"Gail L"` + `"Lugo"` → `"Gail L Lugo"`
- Same API call extracts **ORG** (employer names) and **LOC** (location) at no extra cost

**Phase 2 — Multi-entity extraction** (same single API call):
- `PER` → candidate name
- `ORG` → employer history (filtered: removes tech keywords, job titles, sub-tokens)
- `LOC` → candidate location

All results flow through `extract_all_contact_info()` → `information_extractor.py` → `ranker.py` → Dashboard NER badge + CSV columns.

---

## Scoring

Final score = weighted sum of 4 components, weights vary by detected domain:

| Component | Description |
|---|---|
| Semantic | Cosine similarity: resume header (1500 chars) vs full JD via `all-MiniLM-L6-v2` |
| Skill | 3-layer matched skills / required skills |
| Total Exp | Non-linear: `ratio^1.5` for underqualified, capped at 1.0 for overqualified |
| Role Exp | Same formula, filtered to domain-specific role months |

**Decision thresholds:**

| Score | Decision |
|---|---|
| ≥ 0.78 | Highly Recommended |
| 0.60 – 0.77 | Qualified for Interview |
| 0.42 – 0.59 | Maybe — Review Manually |
| < 0.42 | Reject |

---

## Setup

### 1. Prerequisites

- Python 3.12 or 3.13 (PyTorch does not yet support 3.14)
- [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) installed at `C:\Program Files\poppler-25.11.0\Library\bin`

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set your HuggingFace token (activates BERT NER Layer 0)

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read access, free tier = 1,000 calls/day).

```bash
# Windows PowerShell
$env:HF_API_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"

# Linux / macOS
export HF_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

The system degrades gracefully — without a token, Layers 1–3 handle extraction automatically.

---

## Usage

### Streamlit Dashboard (recommended)

```bash
streamlit run src/Dashboard.py
```

Upload a JD (`.txt` or `.pdf`) + multiple resume PDFs → click **Screen Resumes**.

Features:
- Per-candidate radar chart + score breakdown
- NER source badge per candidate (BERT NER / spaCy / Heuristic)
- BERT-extracted employer names as chips
- Analytics tab: score distribution, NER layer stats, parse quality
- Export to CSV or JSON (includes NER Source, Location, Companies columns)

### CLI — Batch processing

```bash
cd src
python batch_test.py                            # interactive JD input, default resumes/ folder
python batch_test.py --folder path/to/pdfs --jd path/to/jd.txt
```

Output saved to `outputs/results_YYYYMMDD_HHMMSS.{json,csv}`.

### Test a single PDF

```bash
cd src
python test_parser.py
```

### Validate the BERT NER layer

```bash
cd src
python test_ner_bert_layer.py       # Phase 1: name extraction across 5 resumes
python test_ner_multientity.py      # Phase 2: PER + ORG + LOC extraction
```

---

## Project Structure

```
AI_powered_Resume_screening/
├── src/
│   ├── pdf_parser.py           # PDF text extraction (pdfplumber + OCR)
│   ├── ner_extractor.py        # BERT NER + 4-layer name extraction + multi-entity
│   ├── information_extractor.py# Skills, contact info, location, companies
│   ├── experience_extractor.py # Work history parsing, date-range deduplication
│   ├── skill_matcher.py        # 3-layer skill matching with embedding cache
│   ├── matcher.py              # Domain-aware scoring engine
│   ├── domain_config.py        # 10 built-in domain configs + Claude API fallback
│   ├── ranker.py               # Batch pipeline, export
│   ├── Dashboard.py            # Streamlit web UI
│   ├── batch_test.py           # CLI entry point
│   ├── test_parser.py          # Single PDF test
│   ├── test_ner_bert_layer.py  # Phase 1 NER validation
│   └── test_ner_multientity.py # Phase 2 multi-entity validation
├── requirements.txt
├── CLAUDE.md                   # Claude Code project instructions
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| NER Layer 0 | `dslim/bert-base-NER` via HuggingFace Serverless Inference API |
| NER Layer 1 | spaCy `en_core_web_sm` |
| Semantic similarity | `sentence-transformers/all-MiniLM-L6-v2` |
| Fuzzy matching | `rapidfuzz` (threshold 85) |
| PDF parsing | `pdfplumber` + `pdf2image` + `pytesseract` |
| Dashboard | `streamlit` + `plotly` |
| Domain detection | Keyword scoring + Claude API fallback |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_API_TOKEN` | Optional | HuggingFace token — activates BERT NER Layer 0 |

---

## Known Limitations

- **Python 3.14**: PyTorch is not yet supported on Python 3.14. Use Python 3.12 or 3.13. The BERT NER layer works on all versions (API-based, no local torch needed).
- **Tesseract / Poppler paths**: Hardcoded for Windows in `pdf_parser.py:11-12`. Update if your installation paths differ.
- **HuggingFace free tier**: 1,000 API calls/day. The system falls back to spaCy + heuristics when the limit is reached.

---

## Authors

Developed by **Umer** — AI-powered resume screening with novel BERT NER integration.
