import re
import json
import requests
from rapidfuzz import fuzz

# =========================================================
# BUILT-IN DOMAIN CONFIGS (your known categories)
# Easily extensible — just add a new dict entry
# =========================================================
DOMAIN_CONFIGS = {

    "ml_engineer": {
        "keywords": [
            "machine learning engineer", "ml engineer", "ai engineer",
            "mlops", "deep learning", "neural network", "model deployment"
        ],
        "scoring_weights": {
            "semantic": 0.20, "skills": 0.35,
            "total_exp": 0.15, "role_exp": 0.30
        },
        "core_skills": [
            "python", "pytorch", "tensorflow", "scikit-learn",
            "deep learning", "machine learning", "mlops", "transformers",
            "computer vision", "nlp", "docker", "kubernetes"
        ],
        "required_exp_years": 2,
    },

    "data_scientist": {
        "keywords": [
            "data scientist", "data science", "statistical analysis",
            "predictive modeling", "research scientist", "analytics"
        ],
        "scoring_weights": {
            "semantic": 0.25, "skills": 0.30,
            "total_exp": 0.20, "role_exp": 0.25
        },
        "core_skills": [
            "python", "r", "statistics", "machine learning", "pandas",
            "numpy", "tableau", "power bi", "sql", "data analysis",
            "feature engineering", "scikit-learn", "matplotlib"
        ],
        "required_exp_years": 2,
    },

    "data_engineer": {
        "keywords": [
            "data engineer", "etl", "data pipeline", "data warehouse",
            "big data", "apache spark", "airflow", "kafka"
        ],
        "scoring_weights": {
            "semantic": 0.20, "skills": 0.35,
            "total_exp": 0.20, "role_exp": 0.25
        },
        "core_skills": [
            "python", "sql", "apache spark", "kafka", "airflow",
            "etl", "aws", "azure", "gcp", "postgresql",
            "hadoop", "dbt", "data modeling", "docker"
        ],
        "required_exp_years": 2,
    },

    "software_engineer": {
        "keywords": [
            "software engineer", "software developer", "backend",
            "frontend", "full stack", "web developer", "sde", "swe"
        ],
        "scoring_weights": {
            "semantic": 0.20, "skills": 0.40,
            "total_exp": 0.15, "role_exp": 0.25
        },
        "core_skills": [
            "python", "java", "c++", "javascript", "typescript",
            "react", "node.js", "django", "fastapi", "sql",
            "docker", "git", "rest api", "microservices", "aws"
        ],
        "required_exp_years": 2,
    },

    "teacher": {
        "keywords": [
            "teacher", "lecturer", "professor", "instructor",
            "educator", "teaching", "curriculum", "classroom",
            "academic", "school", "college", "university"
        ],
        "scoring_weights": {
            "semantic": 0.35, "skills": 0.20,
            "total_exp": 0.30, "role_exp": 0.15
        },
        "core_skills": [
            "curriculum development", "lesson planning",
            "classroom management", "student assessment",
            "e-learning", "lms", "moodle", "google classroom",
            "differentiated instruction", "pedagogy",
            "communication", "leadership", "research"
        ],
        "required_exp_years": 1,
    },

    "accountant": {
        "keywords": [
            "accountant", "accounting", "financial reporting",
            "taxation", "audit", "bookkeeping", "cpa", "ca",
            "balance sheet", "general ledger"
        ],
        "scoring_weights": {
            "semantic": 0.25, "skills": 0.30,
            "total_exp": 0.25, "role_exp": 0.20
        },
        "core_skills": [
            "accounting", "taxation", "financial reporting", "audit",
            "excel", "quickbooks", "sap", "erp", "bookkeeping",
            "ifrs", "gaap", "payroll", "budgeting", "forecasting"
        ],
        "required_exp_years": 2,
    },

    "doctor": {
        "keywords": [
            "doctor", "physician", "mbbs", "medical officer",
            "clinical", "patient care", "hospital", "surgery",
            "diagnosis", "healthcare"
        ],
        "scoring_weights": {
            "semantic": 0.35, "skills": 0.20,
            "total_exp": 0.30, "role_exp": 0.15
        },
        "core_skills": [
            "patient care", "clinical diagnosis", "surgery",
            "emergency medicine", "pharmacology", "medical ethics",
            "electronic health records", "ehr", "research",
            "communication", "teamwork"
        ],
        "required_exp_years": 1,
    },

    "lawyer": {
        "keywords": [
            "lawyer", "attorney", "advocate", "legal counsel",
            "litigation", "corporate law", "llb", "bar",
            "legal research", "contracts"
        ],
        "scoring_weights": {
            "semantic": 0.35, "skills": 0.25,
            "total_exp": 0.25, "role_exp": 0.15
        },
        "core_skills": [
            "legal research", "litigation", "contract drafting",
            "corporate law", "intellectual property", "compliance",
            "negotiation", "communication", "critical thinking",
            "legal writing", "case management"
        ],
        "required_exp_years": 2,
    },

    "hr_manager": {
        "keywords": [
            "hr", "human resources", "talent acquisition",
            "recruitment", "payroll", "employee relations",
            "performance management", "hris", "onboarding"
        ],
        "scoring_weights": {
            "semantic": 0.30, "skills": 0.25,
            "total_exp": 0.25, "role_exp": 0.20
        },
        "core_skills": [
            "recruitment", "talent acquisition", "payroll",
            "employee relations", "performance management",
            "hris", "onboarding", "training", "labor law",
            "communication", "excel", "conflict resolution"
        ],
        "required_exp_years": 2,
    },

    "graphic_designer": {
        "keywords": [
            "graphic designer", "visual designer", "ui designer",
            "ux designer", "creative designer", "brand designer",
            "adobe", "figma", "illustrator", "photoshop"
        ],
        "scoring_weights": {
            "semantic": 0.30, "skills": 0.35,
            "total_exp": 0.15, "role_exp": 0.20
        },
        "core_skills": [
            "adobe photoshop", "adobe illustrator", "figma",
            "canva", "typography", "branding", "ui design",
            "ux design", "color theory", "indesign",
            "motion graphics", "after effects"
        ],
        "required_exp_years": 1,
    },

    # ── Fallback ──────────────────────────────────────────
    "general": {
        "keywords": [],
        "scoring_weights": {
            "semantic": 0.30, "skills": 0.30,
            "total_exp": 0.20, "role_exp": 0.20
        },
        "core_skills": [],
        "required_exp_years": 1,
    },
}


# =========================================================
# STEP 1: FAST KEYWORD DETECTOR (no LLM, no cost)
# =========================================================
def detect_domain_fast(jd_text: str) -> tuple[str, dict] | None:
    """
    Tries to match JD to a known domain using keyword scoring.
    Returns (domain_name, config) if confident, else None.
    """
    jd_lower = jd_text.lower()
    scores = {}

    for domain, config in DOMAIN_CONFIGS.items():
        if domain == "general":
            continue
        # Count keyword hits + fuzzy partial matches
        hit_count = 0
        for kw in config["keywords"]:
            if kw in jd_lower:
                hit_count += 2          # exact hit — stronger signal
            elif fuzz.partial_ratio(jd_lower[:500], kw) > 85:
                hit_count += 1          # fuzzy hit — weaker signal
        scores[domain] = hit_count

    best_domain = max(scores, key=scores.get)

    # Confident match: at least 2 keyword hits
    if scores[best_domain] >= 2:
        print(f"✅ Domain detected: {best_domain} (score: {scores[best_domain]})")
        return best_domain, DOMAIN_CONFIGS[best_domain]

    return None     # not confident → trigger LLM


# =========================================================
# STEP 2: LLM FALLBACK (only for unknown domains)
# Called ONCE per JD — not per resume
# =========================================================
LLM_PROMPT = """
You are an HR domain classifier. Analyze this job description and return ONLY a JSON object.

Extract:
- domain: short snake_case job category (e.g. "civil_engineer", "nurse", "marketing_manager")
- core_skills: list of 8-12 most important skills for this role
- required_exp_years: minimum years of experience required (integer)
- scoring_weights: object with keys semantic, skills, total_exp, role_exp — all floats summing to 1.0
  (skill-heavy roles: higher skills weight; experience-heavy: higher total_exp weight)

Rules:
- Return ONLY valid JSON, no explanation, no markdown
- All scoring_weights values must sum exactly to 1.0

Job Description:
{jd_text}
"""

def detect_domain_llm(jd_text: str) -> tuple[str, dict]:
    """
    Uses Claude API to extract domain config for unknown job types.
    Falls back to 'general' if API call fails.
    """
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         "YOUR_API_KEY_HERE",
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-haiku-4-5-20251001",   # cheapest — ~$0.001 per JD
                "max_tokens": 500,
                "messages": [{
                    "role":    "user",
                    "content": LLM_PROMPT.format(jd_text=jd_text[:2000])
                }]
            },
            timeout=15
        )

        raw = response.json()["content"][0]["text"].strip()

        # Strip markdown fences if model adds them
        raw = re.sub(r"```json|```", "", raw).strip()
        llm_config = json.loads(raw)

        domain_name = llm_config.get("domain", "general")

        # Build config in same structure as DOMAIN_CONFIGS
        config = {
            "keywords":          [],
            "scoring_weights":   llm_config.get("scoring_weights", {
                "semantic": 0.30, "skills": 0.30,
                "total_exp": 0.20, "role_exp": 0.20
            }),
            "core_skills":       llm_config.get("core_skills", []),
            "required_exp_years": llm_config.get("required_exp_years", 1),
        }

        # Normalize weights — ensure they sum to 1.0
        weights = config["scoring_weights"]
        total   = sum(weights.values())
        if total > 0:
            config["scoring_weights"] = {
                k: round(v / total, 3)
                for k, v in weights.items()
            }

        print(f"🤖 LLM detected domain: {domain_name}")

        # Cache in memory for this session
        DOMAIN_CONFIGS[domain_name] = config
        return domain_name, config

    except Exception as e:
        print(f"⚠ LLM domain detection failed: {e} → using general config")
        return "general", DOMAIN_CONFIGS["general"]


# =========================================================
# MASTER DETECTOR — called once per JD
# =========================================================
def detect_domain(jd_text: str) -> tuple[str, dict]:
    """
    Two-stage domain detection:
    Stage 1 — Fast keyword matching (free, instant)
    Stage 2 — LLM extraction (only if Stage 1 fails)
    """
    # Stage 1: try fast detection first
    result = detect_domain_fast(jd_text)
    if result:
        return result

    # Stage 2: LLM for unknown domains
    print("⚠ Domain unclear from keywords → using LLM...")
    return detect_domain_llm(jd_text)


def get_domain_config(domain_name: str) -> dict:
    return DOMAIN_CONFIGS.get(domain_name, DOMAIN_CONFIGS["general"])