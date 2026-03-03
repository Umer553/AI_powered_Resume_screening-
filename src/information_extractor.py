import re
from ner_extractor import extract_all_contact_info  # ← uses our updated ner_extractor


# =========================================================
# SKILLS DATABASE (expanded per resume category)
# =========================================================
COMMON_SKILLS = {
    # ── Languages ─────────────────────────────────────────
    "python", "java", "c++", "c#", "r", "scala", "julia",
    "javascript", "typescript", "bash", "matlab",

    # ── ML / AI ───────────────────────────────────────────
    "machine learning", "deep learning", "nlp",
    "natural language processing", "computer vision",
    "reinforcement learning", "transfer learning",
    "generative ai", "large language models", "llm",

    # ── Frameworks ────────────────────────────────────────
    "tensorflow", "pytorch", "keras", "scikit-learn",
    "xgboost", "lightgbm", "catboost", "hugging face",
    "transformers", "spacy", "nltk", "opencv",
    "fastapi", "flask", "django", "streamlit",

    # ── Data ──────────────────────────────────────────────
    "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "data analysis", "data science", "data engineering",
    "etl", "feature engineering", "data wrangling",

    # ── Databases ─────────────────────────────────────────
    "sql", "mysql", "postgresql", "mongodb", "redis",
    "sqlite", "oracle", "cassandra", "elasticsearch",

    # ── Cloud / MLOps ─────────────────────────────────────
    "aws", "azure", "gcp", "google cloud",
    "docker", "kubernetes", "mlops", "ci/cd",
    "aws sagemaker", "mlflow", "airflow", "dvc",

    # ── BI / Visualization ────────────────────────────────
    "tableau", "power bi", "excel", "looker", "qlik",

    # ── Big Data ──────────────────────────────────────────
    "apache spark", "hadoop", "kafka", "hive",

    # ── Dev Tools ─────────────────────────────────────────
    "git", "github", "gitlab", "jira", "linux",

    # ── Teaching (for TEACHER category) ───────────────────
    "curriculum development", "lesson planning",
    "classroom management", "e-learning", "lms",
    "moodle", "google classroom", "differentiated instruction",
    "student assessment", "pedagogy",

    # ── Soft Skills (for scoring completeness) ────────────
    "communication", "teamwork", "leadership",
    "problem solving", "critical thinking", "research",
}


def extract_skills(text: str) -> list[str]:
    """
    Skill extraction with case-insensitive matching.
    Handles: TensorFlow, Scikit-Learn, PyTorch etc.
    Also strips version numbers before matching.
    """
    if not text:
        return []

    # Normalize: lowercase + remove version numbers (TensorFlow 2.0 → tensorflow)
    text_lower = re.sub(r'\b\d+\.\d+\b', '', text.lower())

    found = set()
    for skill in COMMON_SKILLS:
        # Word-boundary aware matching (prevents "r" matching "framework")
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)

    return sorted(list(found))


# =========================================================
# EXPERIENCE EXTRACTION
# Moved to experience_extractor.py — only date-range
# based calculation is accurate. Self-reported "X years"
# is kept here as a LAST RESORT fallback only.
# =========================================================
def extract_experience_fallback(text: str) -> float:
    """
    FALLBACK ONLY — used when no date ranges found.
    Extracts first self-reported experience mention.
    Does NOT sum multiple mentions (that was the bug).

    Returns years as float, or 0.0 if nothing found.
    """
    text_lower = text.lower()

    # Only match the FIRST occurrence to avoid summing skill durations
    match = re.search(
        r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:experience|work|professional)',
        text_lower
    )
    if match:
        return float(match.group(1))

    # Months fallback (convert to years)
    match = re.search(
        r'(\d+)\s*\+?\s*months?\s+(?:of\s+)?(?:experience|work)',
        text_lower
    )
    if match:
        return round(int(match.group(1)) / 12, 2)

    return 0.0


# =========================================================
# MASTER FUNCTION
# =========================================================
def extract_candidate_info(text: str, parsed_confidence: float = 1.0) -> dict:
    """
    Orchestrates all extraction.
    Now accepts parsed_confidence from pdf_parser's ParsedDocument
    and passes it through to the final result.

    experience_years here is FALLBACK only.
    Accurate experience is calculated in experience_extractor.py
    from date ranges and passed in separately.
    """
    if not text:
        return _empty_result()

    # ── Contact info (from updated ner_extractor) ─────────
    contact = extract_all_contact_info(text)

    # ── Skills ────────────────────────────────────────────
    skills = extract_skills(text)

    # ── Experience (fallback only) ────────────────────────
    exp_fallback = extract_experience_fallback(text)

    return {
        # Identity
        "name":             contact["name"],
        "name_confidence":  contact["name_confidence"],
        "name_source":      contact["name_source"],

        # Contact
        "email":            contact["email"],
        "phone":            contact["phone"],
        "linkedin":         contact.get("linkedin"),
        "github":           contact.get("github"),

        # Skills
        "skills":           skills,
        "skill_count":      len(skills),

        # Experience (date-range based filled in by experience_extractor later)
        "experience_years_fallback": exp_fallback,
        "experience_years":          None,   # filled by experience_extractor.py
        "role_experience":           {},     # filled by experience_extractor.py

        # Quality signals
        "parse_confidence": parsed_confidence,
        "needs_review":     parsed_confidence < 0.5 or contact["name"] is None,
    }


def _empty_result() -> dict:
    return {
        "name": None, "name_confidence": 0.0, "name_source": "failed",
        "email": None, "phone": None, "linkedin": None, "github": None,
        "skills": [], "skill_count": 0,
        "experience_years_fallback": 0.0, "experience_years": None,
        "role_experience": {}, "parse_confidence": 0.0, "needs_review": True,
    }