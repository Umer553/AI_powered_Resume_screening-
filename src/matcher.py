"""
matcher.py
Computes domain-aware match score between a resume and job description.
Integrates: semantic similarity, skill matching, experience scoring.
"""

import re
import math
from sentence_transformers import SentenceTransformer, util
from skill_matcher import match_skills, extract_jd_skills
from experience_extractor import ExperienceResult
from domain_config import detect_domain

# =========================================================
# SINGLE SHARED MODEL INSTANCE
# Lazy-loaded once — not reloaded per resume
# =========================================================
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


# =========================================================
# SEMANTIC SIMILARITY
# =========================================================
def compute_semantic_score(resume_text: str, jd_text: str) -> float:
    """
    Compares first 1500 chars of resume against full JD.
    Header/summary section carries most semantic signal.
    Returns float 0.0 - 1.0
    """
    model = get_model()
    resume_chunk = resume_text[:1500]
    resume_emb   = model.encode(resume_chunk, convert_to_tensor=True)
    jd_emb       = model.encode(jd_text,      convert_to_tensor=True)
    similarity   = util.cos_sim(resume_emb, jd_emb).item()
    return round(max(0.0, similarity), 4)


# =========================================================
# EXPERIENCE REQUIREMENT EXTRACTOR
# =========================================================
def extract_required_experience(jd_text: str, domain_config: dict = None) -> float:
    """
    Extracts minimum required years from JD text.
    Falls back to domain config default if not found.
    Returns years as float.
    """
    jd_lower = jd_text.lower()

    patterns = [
        r'(?:minimum|at least|over|minimum of)?\s*(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'experience\s+(?:of\s+)?(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:relevant|related|work|professional)',
        r'(\d+)\+?\s*years?',   # last resort: any "X years" mention
    ]

    for pattern in patterns:
        match = re.search(pattern, jd_lower)
        if match:
            return float(match.group(1))

    if domain_config:
        return float(domain_config.get("required_exp_years", 1))

    return 1.0


# =========================================================
# NON-LINEAR EXPERIENCE SCORE
# =========================================================
def experience_score(candidate_months: float, required_months: float) -> float:
    """
    Non-linear scoring:
    - Under-qualified: steep penalty (ratio^1.5)
    - Exactly qualified: ~0.93
    - Overqualified: capped at 1.0, no bonus
    """
    if required_months <= 0:
        return 1.0
    ratio = candidate_months / required_months
    if ratio >= 1.0:
        return min(1.0, 0.9 + 0.1 * math.log(ratio + 1))
    return round(ratio ** 1.5, 4)


# =========================================================
# MASTER SCORING FUNCTION
# =========================================================
def compute_match_score(
    resume_text:        str,
    jd_text:            str,
    candidate_skills:   list,
    exp_result:         ExperienceResult,
    domain_name:        str  = None,
    domain_config:      dict = None,
) -> dict:
    """
    Computes final domain-aware match score.
    All component scores are 0.0 - 1.0.
    Final score is 0.0 - 1.0 (multiply by 100 for display only in ranker).

    Parameters:
        resume_text      : cleaned resume text from pdf_parser
        jd_text          : full job description text
        candidate_skills : list of skills from information_extractor
        exp_result       : ExperienceResult from experience_extractor
        domain_name      : detected domain string e.g. "ml_engineer"
        domain_config    : full config dict from domain_config.py
    """

    # Auto-detect domain if not provided
    if domain_config is None:
        domain_name, domain_config = detect_domain(jd_text)

    weights     = domain_config["scoring_weights"]
    core_skills = set(domain_config.get("core_skills", []))

    # ── 1. Semantic score ─────────────────────────────────
    semantic_score = compute_semantic_score(resume_text, jd_text)

    # ── 2. Skill score (three-layer matching) ─────────────
    jd_skills    = extract_jd_skills(jd_text)
    skill_result = match_skills(
        candidate_skills = candidate_skills,
        jd_skills        = jd_skills,
        domain_config    = domain_config,
    )
    skill_score = skill_result["score"]     # already 0.0 - 1.0

    # ── 3. Total experience score (non-linear) ────────────
    required_months  = extract_required_experience(jd_text, domain_config) * 12
    total_exp_score  = experience_score(exp_result.total_months, required_months)

    # ── 4. Role-specific experience score ─────────────────
    role_months    = exp_result.role_experience.get(domain_name, 0.0)
    role_exp_score = experience_score(role_months, required_months)

    # ── 5. Weighted final score ───────────────────────────
    final_score = (
        weights["semantic"]  * semantic_score  +
        weights["skills"]    * skill_score     +
        weights["total_exp"] * total_exp_score +
        weights["role_exp"]  * role_exp_score
    )
    final_score = round(min(max(final_score, 0.0), 1.0), 4)

    return {
        # Core scores — all 0.0 to 1.0
        "final_score":          final_score,
        "semantic_score":       round(semantic_score,   4),
        "skill_score":          round(skill_score,      4),
        "total_exp_score":      round(total_exp_score,  4),
        "role_exp_score":       round(role_exp_score,   4),

        # Experience detail
        "total_exp_years":      exp_result.total_years,
        "total_exp_months":     exp_result.total_months,
        "role_exp_months":      role_months,
        "required_exp_years":   required_months / 12,

        # Skill breakdown
        "matched_skills":       skill_result.get("matched_exact", []) +
                                skill_result.get("matched_fuzzy", []),
        "semantic_skills":      skill_result.get("matched_semantic", []),
        "missing_skills":       skill_result.get("missing_skills", []),
        "skill_breakdown":      skill_result.get("match_breakdown", {}),

        # Domain metadata
        "domain":               domain_name,
        "scoring_weights_used": weights,
    }


# =========================================================
# HIRING DECISION
# =========================================================
def hiring_decision(final_score: float) -> str:
    """
    Thresholds on 0.0 - 1.0 scale.
    Score formula already accounts for experience via non-linear function.
    """
    if final_score >= 0.78:
        return "⭐ Highly Recommended — Strong Fit"
    elif final_score >= 0.60:
        return "✅ Qualified for Interview"
    elif final_score >= 0.42:
        return "🟡 Maybe — Review Manually"
    else:
        return "❌ Reject"


# =========================================================
# STANDALONE TEST
# =========================================================
if __name__ == "__main__":
    from pdf_parser            import extract_text_from_pdf
    from information_extractor import extract_candidate_info
    from experience_extractor  import extract_work_experience
    from domain_config         import detect_domain

    pdf_path = r"C:\Aqib_project\AI_powered_Resume_screening\resumes\data\ML_ENGINEER\sample.pdf"
    jd_text  = """
    We are looking for a Machine Learning Engineer with 3+ years of experience.
    Must have strong Python, PyTorch or TensorFlow skills.
    Experience with MLOps, Docker, and model deployment required.
    NLP and computer vision experience is a plus.
    """

    parsed      = extract_text_from_pdf(pdf_path)
    info        = extract_candidate_info(parsed.text, parsed.confidence)
    exp_result  = extract_work_experience(parsed.text)
    domain_name, domain_config = detect_domain(jd_text)

    result = compute_match_score(
        resume_text      = parsed.text,
        jd_text          = jd_text,
        candidate_skills = info["skills"],
        exp_result       = exp_result,
        domain_name      = domain_name,
        domain_config    = domain_config,
    )

    print(f"\nDomain          : {result['domain']}")
    print(f"Final Score     : {result['final_score'] * 100:.1f}%")
    print(f"Semantic        : {result['semantic_score'] * 100:.1f}%")
    print(f"Skill Score     : {result['skill_score'] * 100:.1f}%")
    print(f"Total Exp Score : {result['total_exp_score'] * 100:.1f}%")
    print(f"Role Exp Score  : {result['role_exp_score'] * 100:.1f}%")
    print(f"Matched Skills  : {result['matched_skills']}")
    print(f"Missing Skills  : {result['missing_skills']}")
    print(f"Decision        : {hiring_decision(result['final_score'])}")