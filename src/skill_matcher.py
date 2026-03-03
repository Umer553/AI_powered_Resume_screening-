"""
skill_matcher.py
Three-layer skill matching: exact → fuzzy → semantic.
Domain-aware: boosts core skills based on job type.
"""

import re
import hashlib
import pickle
from pathlib import Path
from rapidfuzz import fuzz, process

# =========================================================
# EMBEDDING CACHE SETUP
# model is NOT loaded here — imported from matcher.py
# to avoid loading the 90MB model twice
# =========================================================
CACHE_DIR = Path("cache/skill_embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cached_embedding(text: str):
    """
    Cache embeddings by text hash.
    Avoids re-encoding the same skill word 1000x during batch.
    Uses the shared model instance from matcher.py.
    """
    key        = hashlib.md5(text.lower().encode()).hexdigest()
    cache_path = CACHE_DIR / f"{key}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Import shared model — not reloaded if already loaded
    from matcher import get_model
    model     = get_model()
    embedding = model.encode(text, convert_to_tensor=True)

    with open(cache_path, "wb") as f:
        pickle.dump(embedding, f)

    return embedding


# =========================================================
# SKILL NORMALIZATION
# =========================================================
SKILL_ALIASES = {
    "scikit learn":               "scikit-learn",
    "sklearn":                    "scikit-learn",
    "sk-learn":                   "scikit-learn",
    "tensor flow":                "tensorflow",
    "py torch":                   "pytorch",
    "natural language processing":"nlp",
    "cv":                         "computer vision",
    "powerbi":                    "power bi",
    "ms excel":                   "excel",
    "node":                       "node.js",
    "nodejs":                     "node.js",
    "postgres":                   "postgresql",
    "aws sagemaker":              "sagemaker",
    "huggingface":                "hugging face",
    "hf":                         "hugging face",
    "gpt":                        "large language models",
    "llms":                       "large language models",
}

def normalize_skill(skill: str) -> str:
    """Lowercase, strip version numbers, resolve aliases."""
    cleaned = skill.lower().strip()
    cleaned = re.sub(r'\b\d+\.\d+\b', '', cleaned).strip()
    return SKILL_ALIASES.get(cleaned, cleaned)


# =========================================================
# JD SKILL EXTRACTOR
# =========================================================
from information_extractor import extract_skills

def extract_jd_skills(jd_text: str) -> set:
    """
    Extracts required skills from job description.
    Reuses same skill database as resume extraction —
    both sides use identical vocabulary for fair matching.
    """
    return set(extract_skills(jd_text))


# =========================================================
# SKILL IMPLICATION GRAPH
# =========================================================
SKILL_IMPLIES = {
    "tensorflow":        {"python", "numpy", "keras", "deep learning"},
    "pytorch":           {"python", "numpy", "deep learning"},
    "scikit-learn":      {"python", "numpy", "pandas", "machine learning"},
    "pandas":            {"python", "numpy"},
    "keras":             {"python", "deep learning", "tensorflow"},
    "fastapi":           {"python", "rest api"},
    "django":            {"python", "sql", "rest api"},
    "flask":             {"python", "rest api"},
    "apache spark":      {"big data", "distributed computing"},
    "kubernetes":        {"docker", "devops"},
    "sagemaker":         {"aws", "cloud", "machine learning"},
    "dbt":               {"sql", "data modeling"},
    "opencv":            {"python", "computer vision"},
    "hugging face":      {"python", "nlp", "transformers"},
    "react":             {"javascript", "html", "css"},
    "postgresql":        {"sql"},
    "mysql":             {"sql"},
    "mongodb":           {"nosql"},
    "large language models": {"python", "nlp", "transformers"},
    "mlflow":            {"python", "mlops", "machine learning"},
    "airflow":           {"python", "etl", "data engineering"},
}

def expand_skills(skills: set) -> set:
    """Add implied skills to a skill set."""
    expanded = set(skills)
    for skill in list(skills):
        expanded.update(SKILL_IMPLIES.get(normalize_skill(skill), set()))
    return expanded


# =========================================================
# THREE-LAYER SKILL MATCHING
# =========================================================
def match_skills(
    candidate_skills:   list,
    jd_skills:          set,
    domain_config:      dict = None,     # ← ADDED: domain awareness
    fuzzy_threshold:    int  = 85,
    semantic_threshold: float = 0.72,
) -> dict:
    """
    Three-layer matching pipeline:

    Layer 1 — Exact match (after normalization + alias resolution)
    Layer 2 — Fuzzy match (handles typos, spacing: scikit learn vs scikit-learn)
    Layer 3 — Semantic match (handles synonyms: NLP vs natural language processing)

    Domain-aware: core skills for the detected domain get a score boost.
    Returns normalized score 0.0–1.0 plus full breakdown for dashboard.
    """
    if not jd_skills:
        return _empty_match_result()

    # Get domain core skills for boosting (empty set if no domain)
    core_skills = set(domain_config.get("core_skills", [])) if domain_config else set()

    # Normalize both skill sets
    candidate_normalized = {normalize_skill(s) for s in candidate_skills}
    jd_normalized        = {normalize_skill(s) for s in jd_skills}

    # Expand with implied skills
    candidate_expanded = expand_skills(candidate_normalized)
    jd_expanded        = expand_skills(jd_normalized)

    matched_exact    = []
    matched_fuzzy    = []
    matched_semantic = []
    missing          = []

    from sentence_transformers import util as st_util

    for jd_skill in jd_expanded:

        # ── Layer 1: Exact match ──────────────────────────
        if jd_skill in candidate_expanded:
            matched_exact.append(jd_skill)
            continue

        # ── Layer 2: Fuzzy match ──────────────────────────
        result = process.extractOne(
            jd_skill,
            candidate_expanded,
            scorer=fuzz.token_sort_ratio
        )
        if result and result[1] >= fuzzy_threshold:
            matched_fuzzy.append(jd_skill)
            continue

        # ── Layer 3: Semantic match ───────────────────────
        # Skill-to-skill comparison (not skill-to-full-JD)
        jd_emb        = get_cached_embedding(jd_skill)
        best_semantic = 0.0

        for cand_skill in candidate_expanded:
            cand_emb = get_cached_embedding(cand_skill)
            sim      = st_util.cos_sim(jd_emb, cand_emb).item()
            if sim > best_semantic:
                best_semantic = sim

        if best_semantic >= semantic_threshold:
            matched_semantic.append(jd_skill)
        else:
            missing.append(jd_skill)

    # ── Domain-aware weighted score ───────────────────────
    # Core skills for the detected domain get a 20% boost
    total_jd_skills  = len(jd_expanded)
    weighted_matched = 0.0

    for skill in matched_exact:
        weighted_matched += 1.20 if skill in core_skills else 1.00

    for skill in matched_fuzzy:
        weighted_matched += 1.10 if skill in core_skills else 0.90

    for skill in matched_semantic:
        weighted_matched += 0.85 if skill in core_skills else 0.75

    score = round(weighted_matched / total_jd_skills, 4) if total_jd_skills > 0 else 0.0
    score = min(score, 1.0)     # cap at 1.0 — core skill boosts can push above 1

    return {
        "score":            score,
        "matched_exact":    matched_exact,
        "matched_fuzzy":    matched_fuzzy,
        "matched_semantic": matched_semantic,
        "missing_skills":   missing,
        "total_jd_skills":  total_jd_skills,
        "total_matched":    len(matched_exact) + len(matched_fuzzy) + len(matched_semantic),
        "match_breakdown": {
            "exact":    len(matched_exact),
            "fuzzy":    len(matched_fuzzy),
            "semantic": len(matched_semantic),
            "missing":  len(missing),
        }
    }


def _empty_match_result() -> dict:
    return {
        "score": 0.0, "matched_exact": [], "matched_fuzzy": [],
        "matched_semantic": [], "missing_skills": [],
        "total_jd_skills": 0, "total_matched": 0, "match_breakdown": {}
    }


# =========================================================
# STANDALONE TEST
# =========================================================
if __name__ == "__main__":
    candidate_skills = [
        "Python", "TensorFlow 2.0", "Scikit-Learn",
        "Pandas", "Docker", "Deep Learning", "NLP"
    ]
    jd_text = """
    We are looking for an ML Engineer with strong Python skills.
    Experience with PyTorch or TensorFlow required.
    Knowledge of scikit-learn, numpy, and MLOps practices preferred.
    Docker and NLP experience is a plus.
    """

    jd_skills = extract_jd_skills(jd_text)
    print("JD Skills Extracted:", jd_skills)

    result = match_skills(candidate_skills, jd_skills)

    print(f"\nSkill Score      : {result['score']}")
    print(f"Exact Matches    : {result['matched_exact']}")
    print(f"Fuzzy Matches    : {result['matched_fuzzy']}")
    print(f"Semantic Matches : {result['matched_semantic']}")
    print(f"Missing Skills   : {result['missing_skills']}")
    print(f"Breakdown        : {result['match_breakdown']}")