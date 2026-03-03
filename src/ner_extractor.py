import spacy
import re
from dataclasses import dataclass

nlp = spacy.load("en_core_web_sm")


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
    "spacy_ner":      0.90,
    "first_line":     0.75,
    "regex_fallback": 0.60,
    "failed":         0.00,
}


@dataclass
class ExtractedName:
    value: str | None
    confidence: float
    source: str       # "spacy_ner" | "first_line" | "regex_fallback" | "failed"


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
def extract_name_spacy(header_text: str) -> str | None:
    """Primary: spaCy NER on resume header."""
    doc = nlp(header_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            candidate = ent.text.strip()
            if is_valid_name(candidate):
                return candidate

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
    Three-layer name extraction with confidence scoring.
    Layer 1: spaCy NER (most reliable)
    Layer 2: First-line heuristic (reliable for clean resumes)
    Layer 3: Regex capitalized pattern (last resort)
    """
    if not text:
        return ExtractedName(value=None, confidence=0.0, source="failed")

    # Only scan top of resume
    header_text = "\n".join(text.split("\n")[:30])

    # ── Layer 1: spaCy NER ───────────────────────────────
    name = extract_name_spacy(header_text)
    if name:
        return ExtractedName(
            value=name.title(),
            confidence=NAME_CONFIDENCE["spacy_ner"],
            source="spacy_ner"
        )

    # ── Layer 2: First-line heuristic ────────────────────
    name = extract_name_first_line(text)
    if name:
        return ExtractedName(
            value=name.title(),
            confidence=NAME_CONFIDENCE["first_line"],
            source="first_line"
        )

    # ── Layer 3: Regex fallback ───────────────────────────
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
    Single call to extract all contact fields.
    Used by information_extractor.py
    """
    name_result = extract_name_ner(text)

    return {
        "name":            name_result.value,
        "name_confidence": name_result.confidence,
        "name_source":     name_result.source,
        "email":           extract_email(text),
        "phone":           extract_phone(text),
        "linkedin":        extract_linkedin(text),
        "github":          extract_github(text),
    }