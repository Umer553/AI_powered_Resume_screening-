import os
import re
import requests
from dataclasses import dataclass

# ── HuggingFace Serverless Inference API ─────────────────────────────────────
# Model : dslim/bert-base-NER (BERT-base fine-tuned on CoNLL-2003)
# Why   : Dramatically better PERSON detection than spaCy en_core_web_sm,
#          especially for non-Western names — no local GPU or large RAM needed.
# Auth  : Requires a FREE HuggingFace token (huggingface.co → Settings → Tokens)
#          Set the env var:  HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
#          Free tier: 1,000 API calls/day.  Falls back to spaCy when not set.
_HF_NER_URL = "https://router.huggingface.co/hf-inference/models/dslim/bert-base-NER"
_HF_TOKEN   = os.environ.get("HF_API_TOKEN", "")
_HF_HEADERS = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}

# spaCy model — lazy-loaded only when BERT API is unavailable
_nlp = None


def get_nlp():
    """Lazy-loads spaCy en_core_web_sm on first use (fallback layer)."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ── ORG false-positive filter ────────────────────────────────────
# BERT tags many tech keywords as ORG (TensorFlow, PyTorch, AWS…).
# These are skills, not employers — exclude them from company extraction.
ORG_REJECT = {
    # Tech tools / frameworks
    "python", "tensorflow", "pytorch", "keras", "opencv", "numpy", "pandas",
    "matplotlib", "seaborn", "docker", "git", "fastapi", "flask", "sql",
    "nlp", "mlops", "transformers", "hugging", "face", "scikit", "learn",
    "javascript", "typescript", "java", "scala", "rust", "golang",
    "labview", "dasylab", "testing", "quickbooks", "peachtree",
    # Cloud / infra
    "aws", "azure", "gcp", "google", "cloud", "kubernetes", "linux",
    # BI tools
    "tableau", "excel", "powerbi", "spark", "hadoop", "kafka",
    # Generic org words (not company names by themselves)
    "company", "name", "inc", "llc", "ltd", "corp", "group",
    "applied", "science", "technology", "technologies", "systems",
    # Academic
    "university", "college", "institute", "school", "academy",
}

# ── Rejection blocklist (expanded) ───────────────────────────────
REJECT_WORDS = {
    # Technical skills
    "python", "tensorflow", "pytorch", "keras", "opencv", "numpy", "pandas",
    "matplotlib", "seaborn", "docker", "git", "fastapi", "flask", "sql",
    "nlp", "mlops", "transformers", "machine", "learning", "deep", "data",
    "analysis", "science", "html", "css", "javascript", "java", "scala",
    # Resume section headers
    "skills", "experience", "education", "projects", "summary", "profile",
    "objective", "contact", "references", "certifications", "achievements",
    "languages", "interests", "hobbies", "awards", "publications",
    # Job titles (spaCy confuses these with names)
    "engineer", "developer", "scientist", "analyst", "manager", "director",
    "consultant", "architect", "intern", "lead", "senior", "junior",
    # Institutions / generic
    "university", "institute", "college", "school", "certified", "bachelor",
    "master", "degree", "gpa", "cgpa",
    # Common false positives
    "resume", "curriculum", "vitae", "page", "phone", "email", "github",
    "linkedin", "address", "nationality", "present", "current"
}


# ── Confidence levels for name extraction ────────────────────────
NAME_CONFIDENCE = {
    "bert_ner":       0.95,   # dslim/bert-base-NER (pretrained, highest accuracy)
    "spacy_ner":      0.90,
    "first_line":     0.75,
    "regex_fallback": 0.60,
    "failed":         0.00,
}


@dataclass
class ExtractedName:
    value: str | None
    confidence: float
    source: str       # "bert_ner" | "spacy_ner" | "first_line" | "regex_fallback" | "failed"


# =========================================================
# VALIDATION
# =========================================================
def is_valid_name(name: str) -> bool:
    """
    Validate if a string looks like a real human name.
    Fixed: allows 1-4 words, hyphens, South Asian name patterns.
    """
    if not name:
        return False

    name = name.strip()

    # Reject if contains digits
    if any(char.isdigit() for char in name):
        return False

    words = name.split()

    # ── FIX: expanded range 2-3 → 1-4 ───────────────────
    # Handles: single names, hyphenated, South Asian 3-4 part names
    if not (1 <= len(words) <= 4):
        return False

    # Reject if any word is a known non-name
    for w in words:
        if w.lower().strip("-") in REJECT_WORDS:
            return False

    # Each word should start with uppercase
    # ── FIX: handle hyphenated parts (Ahmad-Raza) ────────
    for w in words:
        parts = w.split("-")
        if not all(p and p[0].isupper() for p in parts):
            return False

    # ── FIX: allow hyphens in names ──────────────────────
    # Original regex rejected Mary-Jane, Ahmad-Raza
    if re.search(r"[^a-zA-Z\s\-]", name):
        return False

    # Reject all-uppercase words (headers like "JOHN SMITH" from bad OCR)
    # Allow it only if it's a short name that can be title-cased safely
    if any(w.isupper() and len(w) > 1 for w in words):
        # Allow but normalize — don't reject
        pass

    return True


# =========================================================
# EXTRACTION METHODS
# =========================================================
def _bert_call(text_chunk: str) -> list:
    """Single HF API call. Returns entity list or [] on any failure."""
    try:
        payload = {
            "inputs": text_chunk,
            "parameters": {"aggregation_strategy": "simple"},
        }
        resp = requests.post(_HF_NER_URL, headers=_HF_HEADERS,
                             json=payload, timeout=8)
        if resp.status_code != 200:
            return []
        entities = resp.json()
        if entities and isinstance(entities[0], list):
            entities = entities[0]
        return entities if isinstance(entities, list) else []
    except Exception:
        return []


def _merge_per_entities(entities: list) -> str | None:
    """
    Merge consecutive PER tokens into one name.
    BERT sometimes splits 'Gail L. Lugo' into ['Gail L', 'Lugo'] —
    adjacent PER entities within 5 chars of each other are joined.
    """
    per = [e for e in entities
           if e.get("entity_group") == "PER" and e.get("score", 0) >= 0.85]
    if not per:
        return None

    per.sort(key=lambda e: e.get("start", 0))

    groups = []
    current = [per[0]]
    for ent in per[1:]:
        prev_end = current[-1].get("end", 0)
        curr_start = ent.get("start", 0)
        if curr_start - prev_end <= 5:
            current.append(ent)
        else:
            groups.append(current)
            current = [ent]
    groups.append(current)

    best = max(groups, key=lambda g: sum(e["score"] for e in g) / len(g))
    merged = " ".join(e.get("word", "").strip() for e in best)
    return merged.strip() if merged.strip() else None


def extract_name_bert(text: str) -> str | None:
    """
    Layer 0 (primary): HuggingFace Serverless Inference API → dslim/bert-base-NER.

    Scans the resume in overlapping 512-char windows (up to 3 windows):
    - Window 1: chars 0-512    (header — catches well-formatted CVs)
    - Window 2: chars 400-912  (overlap — catches names just past header)
    - Window 3: chars 1600-2112 (mid-doc — catches names in dense layouts)

    Adjacent PER tokens are merged so 'Gail L' + 'Lugo' -> 'Gail L Lugo'.
    Falls back silently when token absent or on any network/API error.
    """
    if not _HF_TOKEN:
        return None

    # Four overlapping windows cover the first ~3000 chars of any resume.
    # Most names appear in window 0-1; windows 2-3 catch dense/anonymized layouts.
    windows = [
        text[0:512],
        text[400:912],
        text[1400:1912],
        text[2000:2512],
    ]

    for chunk in windows:
        if not chunk.strip():
            continue
        entities = _bert_call(chunk)
        name = _merge_per_entities(entities)
        if name and is_valid_name(name):
            return name.title()

    return None


# Job-role words that BERT tags as ORG when they appear near company context
_ORG_TITLE_WORDS = {
    "underwriter", "technician", "engineer", "manager", "manage", "management",
    "analyst", "supervisor", "coordinator", "specialist", "consultant", "director",
    "advisor", "associate", "officer", "representative", "clerk",
    "accountant", "developer", "scientist", "architect", "intern",
}


def _extract_orgs(entities: list) -> list[str]:
    """
    Collect ORG entities from a BERT response, filtering out noise:
    - Tech keywords misclassified as ORG (TensorFlow, AWS, etc.)
    - WordPiece sub-tokens that start with '##'
    - Job title phrases (Auto Underwriter, Lab Manager, etc.)
    - Very short strings or all-caps acronyms under 5 chars
    - Overly long generic phrases (> 4 words)
    Returns a deduplicated list of likely employer names.
    """
    seen = set()
    orgs = []
    for ent in entities:
        if ent.get("entity_group") != "ORG":
            continue
        if ent.get("score", 0) < 0.85:
            continue
        name = ent.get("word", "").strip()

        # Drop WordPiece sub-tokens
        if name.startswith("##") or "##" in name:
            continue
        # Too short (acronyms like ACH, ERO misfire often)
        if len(name) < 5:
            continue
        # Too many words — likely a generic phrase, not a company name
        if len(name.split()) > 4:
            continue
        # Filter known tech/skill terms
        if any(reject in name.lower() for reject in ORG_REJECT):
            continue
        # Filter job-title phrases (e.g. "Auto Underwriter", "Lab Manager")
        if any(title in name.lower() for title in _ORG_TITLE_WORDS):
            continue

        key = name.lower()
        if key not in seen:
            seen.add(key)
            orgs.append(name)
    return orgs


def _extract_location(entities: list) -> str | None:
    """
    Extract the first high-confidence LOC entity from a BERT response.
    Used for candidate city/country from the resume header.
    """
    for ent in entities:
        if ent.get("entity_group") == "LOC" and ent.get("score", 0) >= 0.80:
            loc = ent.get("word", "").strip()
            if loc and len(loc) >= 2:
                return loc.title()
    return None


def extract_entities_bert(text: str) -> dict:
    """
    Phase 2: Multi-entity BERT extraction in one pass.
    Returns PER (candidate name), ORGs (employer list), and LOC (location).

    Strategy per entity type:
    - PER  : stop at first window that yields a valid name (same as extract_name_bert)
    - ORG  : scan ALL windows and accumulate unique employers across work history
    - LOC  : take first match from early windows (location is in the header)

    Uses the same 4 sliding windows as the PER extractor.
    Max 4 API calls per resume — all results reused, no redundant calls.
    """
    if not _HF_TOKEN:
        return {"per": None, "orgs": [], "location": None}

    windows = [
        text[0:512],
        text[400:912],
        text[1400:1912],
        text[2000:2512],
    ]

    per_name  = None
    all_orgs  = []
    location  = None
    seen_orgs = set()

    for chunk in windows:
        if not chunk.strip():
            continue

        entities = _bert_call(chunk)

        # PER — stop scanning once found
        if per_name is None:
            candidate = _merge_per_entities(entities)
            if candidate and is_valid_name(candidate):
                per_name = candidate.title()

        # ORG — accumulate from all windows (employers span entire document)
        for org in _extract_orgs(entities):
            if org.lower() not in seen_orgs:
                seen_orgs.add(org.lower())
                all_orgs.append(org)

        # LOC — first confident match from early windows
        if location is None:
            location = _extract_location(entities)

    return {
        "per":      per_name,
        "orgs":     all_orgs[:5],   # cap at 5 most prominent employers
        "location": location,
    }


def extract_name_spacy(header_text: str) -> str | None:
    """Fallback layer 1: spaCy NER on resume header."""
    try:
        doc = get_nlp()(header_text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                candidate = ent.text.strip()
                if is_valid_name(candidate):
                    return candidate
    except Exception:
        pass
    return None


def extract_name_first_line(text: str) -> str | None:
    """
    Fallback 1: Most resumes put the candidate name on the
    very first non-empty, non-email, non-phone line.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    for line in lines[:5]:  # check first 5 non-empty lines
        # Skip lines that look like contact info
        if re.search(r"[\@\|\/\d]", line):
            continue
        # Skip lines that are clearly section headers
        if line.isupper() and len(line.split()) > 2:
            continue
        # Skip lines with common resume keywords
        words_in_line = line.lower().split()
        if any(w in REJECT_WORDS for w in words_in_line):
            continue
        # If remaining line looks like a name
        if is_valid_name(line):
            return line

    return None


def extract_name_regex(header_text: str) -> str | None:
    """
    Fallback 2: Regex pattern for capitalized word sequences.
    Catches names spaCy misses entirely.
    """
    # Pattern: 2-4 capitalized words (allows hyphen)
    pattern = r"\b([A-Z][a-zA-Z\-]+(?:\s[A-Z][a-zA-Z\-]+){1,3})\b"
    matches = re.findall(pattern, header_text)

    for match in matches:
        if is_valid_name(match):
            # Extra check: not a job title phrase
            lower = match.lower()
            if not any(rw in lower for rw in REJECT_WORDS):
                return match

    return None


# =========================================================
# CONTACT INFO EXTRACTORS (unchanged, kept here for cohesion)
# =========================================================
def extract_email(text: str) -> str | None:
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:
    # Handles: +92-300-1234567, (021) 1234567, 0300 1234567 etc.
    match = re.search(
        r"(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)(\d{3,4}[\s\-]?\d{4})",
        text
    )
    return match.group(0).strip() if match else None


def extract_linkedin(text: str) -> str | None:
    match = re.search(r"linkedin\.com/in/[a-zA-Z0-9\-\_]+", text, re.IGNORECASE)
    return match.group(0) if match else None


def extract_github(text: str) -> str | None:
    match = re.search(r"github\.com/[a-zA-Z0-9\-\_]+", text, re.IGNORECASE)
    return match.group(0) if match else None


# =========================================================
# MAIN EXTRACTOR — returns structured result with confidence
# =========================================================
def extract_name_ner(text: str) -> ExtractedName:
    """
    Four-layer name extraction with confidence scoring.
    Layer 0: dslim/bert-base-NER pretrained BERT model (confidence 0.95)
    Layer 1: spaCy NER — en_core_web_sm (confidence 0.90)
    Layer 2: First-line heuristic (confidence 0.75)
    Layer 3: Regex capitalized pattern — last resort (confidence 0.60)
    """
    if not text:
        return ExtractedName(value=None, confidence=0.0, source="failed")

    header_text = "\n".join(text.split("\n")[:30])

    # ── Layer 0: BERT NER — full text, sliding window ────────────
    name = extract_name_bert(text)
    if name:
        return ExtractedName(
            value=name,
            confidence=NAME_CONFIDENCE["bert_ner"],
            source="bert_ner"
        )

    # ── Layer 1: spaCy NER ───────────────────────────────────────
    name = extract_name_spacy(header_text)
    if name:
        return ExtractedName(
            value=name.title(),
            confidence=NAME_CONFIDENCE["spacy_ner"],
            source="spacy_ner"
        )

    # ── Layer 2: First-line heuristic ────────────────────────────
    name = extract_name_first_line(text)
    if name:
        return ExtractedName(
            value=name.title(),
            confidence=NAME_CONFIDENCE["first_line"],
            source="first_line"
        )

    # ── Layer 3: Regex fallback ───────────────────────────────────
    name = extract_name_regex(header_text)
    if name:
        return ExtractedName(
            value=name.title(),
            confidence=NAME_CONFIDENCE["regex_fallback"],
            source="regex_fallback"
        )

    return ExtractedName(value=None, confidence=0.0, source="failed")


def extract_all_contact_info(text: str) -> dict:
    """
    Single call to extract all contact fields + BERT multi-entity results.
    Used by information_extractor.py.

    When HF_API_TOKEN is set, runs extract_entities_bert() once and reuses
    the PER result for name extraction (avoids a redundant API call).
    Falls back to the 4-layer heuristic pipeline when token is absent.
    """
    if _HF_TOKEN:
        # One multi-entity BERT call covers PER + ORG + LOC together
        bert_result = extract_entities_bert(text)
        per_name = bert_result["per"]

        if per_name:
            name_result = ExtractedName(
                value=per_name,
                confidence=NAME_CONFIDENCE["bert_ner"],
                source="bert_ner",
            )
        else:
            # BERT found no name — fall through to spaCy + heuristics
            name_result = _fallback_name_extraction(text)
    else:
        bert_result = {"per": None, "orgs": [], "location": None}
        name_result = _fallback_name_extraction(text)

    return {
        "name":            name_result.value,
        "name_confidence": name_result.confidence,
        "name_source":     name_result.source,
        "email":           extract_email(text),
        "phone":           extract_phone(text),
        "linkedin":        extract_linkedin(text),
        "github":          extract_github(text),
        # Phase 2: new BERT-extracted fields
        "location":        bert_result["location"],
        "companies":       bert_result["orgs"],
    }


def _fallback_name_extraction(text: str) -> ExtractedName:
    """Layers 1-3 when BERT is unavailable or found no name."""
    header_text = "\n".join(text.split("\n")[:30])

    name = extract_name_spacy(header_text)
    if name:
        return ExtractedName(value=name.title(), confidence=NAME_CONFIDENCE["spacy_ner"], source="spacy_ner")

    name = extract_name_first_line(text)
    if name:
        return ExtractedName(value=name.title(), confidence=NAME_CONFIDENCE["first_line"], source="first_line")

    name = extract_name_regex(header_text)
    if name:
        return ExtractedName(value=name.title(), confidence=NAME_CONFIDENCE["regex_fallback"], source="regex_fallback")

    return ExtractedName(value=None, confidence=0.0, source="failed")