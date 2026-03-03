import re
import math
from datetime import date, datetime
from rapidfuzz import fuzz
from dateutil import parser as dateutil_parser
from dataclasses import dataclass, field

# =========================================================
# MONTH MAP
# =========================================================
MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12
}

# =========================================================
# DATE RANGE PATTERNS (ordered: most specific → least)
# =========================================================
DATE_RANGE_PATTERNS = [
    # Aug 2022 - Present | January 2020 – Current
    r'([A-Za-z]{3,9}\.?\s?\d{4})\s*(?:-|–|—|to|until)\s*([A-Za-z]{3,9}\.?\s?\d{4}|[A-Za-z]{3,9}\.?\s?\d{2}|Present|Current|Now|Ongoing)',
    # Aug'22 - Present
    r"([A-Za-z]{3,9}[''`]\d{2})\s*(?:-|–|—|to)\s*([A-Za-z]{3,9}[''`]\d{2}|Present|Current|Now)",
    # 01/2020 - 06/2022
    r'(\d{1,2}/\d{4})\s*(?:-|–|—|to)\s*(\d{1,2}/\d{4}|Present|Current|Now)',
    # 2020 - 2023 | 2020 to 2023
    r'\b(\d{4})\s*(?:-|–|—|to|until)\s*(\d{4}|Present|Current|Now)\b',
]

# =========================================================
# JOB KEYWORDS (expanded for all resume categories)
# =========================================================
JOB_KEYWORDS = [
    # Engineering / Tech
    "engineer", "developer", "architect", "programmer", "devops",
    # Data / ML
    "analyst", "scientist", "researcher", "specialist",
    # Management
    "manager", "director", "lead", "head", "officer", "coordinator",
    # General
    "consultant", "assistant", "associate", "executive", "advisor",
    # Teaching (TEACHER category)
    "teacher", "lecturer", "professor", "instructor", "tutor", "educator",
    # Intern
    "intern", "trainee", "apprentice",
]

# =========================================================
# ROLE TAXONOMY — maps title → canonical role
# =========================================================
ROLE_TAXONOMY = {
    "ml_engineer": [
        "machine learning engineer", "ml engineer", "ai engineer",
        "applied scientist", "mlops engineer", "ai developer",
        "deep learning engineer"
    ],
    "data_scientist": [
        "data scientist", "research scientist", "data analyst",
        "analytics engineer", "quantitative analyst", "statistician"
    ],
    "data_engineer": [
        "data engineer", "etl developer", "pipeline engineer",
        "bigdata engineer", "analytics engineer"
    ],
    "software_engineer": [
        "software engineer", "software developer", "backend engineer",
        "full stack developer", "sde", "swe", "programmer",
        "frontend developer", "web developer"
    ],
    "teacher": [
        "teacher", "lecturer", "professor", "instructor",
        "tutor", "educator", "teaching assistant"
    ],
}

def get_canonical_role(title: str) -> str | None:
    title_lower = title.lower().strip()
    for canonical, aliases in ROLE_TAXONOMY.items():
        if any(alias in title_lower for alias in aliases):
            return canonical
    return None


# =========================================================
# DATE PARSER
# =========================================================
PRESENT_TOKENS = {"present", "current", "now", "ongoing", "till date", "today"}

def parse_date(date_str: str) -> date | None:
    """
    Robust date parser with multiple fallback layers.
    Returns date object or None.
    """
    if not date_str:
        return None

    cleaned = date_str.strip().lower().replace('.', '').replace(',', '')

    # Present tokens
    if cleaned in PRESENT_TOKENS:
        return date.today()

    # Skip noise
    if "month" in cleaned or "year" in cleaned:
        return None

    # ── Layer 1: apostrophe year (Aug'22) ────────────────
    apos_match = re.match(r"([a-z]{3,9})[''`](\d{2})$", cleaned)
    if apos_match:
        mon = MONTH_MAP.get(apos_match.group(1)[:3])
        if mon:
            return date(2000 + int(apos_match.group(2)), mon, 1)

    # ── Layer 2: MM/YYYY ──────────────────────────────────
    slash_match = re.match(r"(\d{1,2})/(\d{4})$", cleaned)
    if slash_match:
        return date(int(slash_match.group(2)), int(slash_match.group(1)), 1)

    # ── Layer 3: Month YYYY (Jan 2022, January 2022) ──────
    parts = cleaned.split()
    if len(parts) == 2:
        mon = MONTH_MAP.get(parts[0][:3])
        if mon and re.match(r'\d{4}', parts[1]):
            return date(int(parts[1]), mon, 1)

    # ── Layer 4: Year only ────────────────────────────────
    if re.match(r'^\d{4}$', cleaned):
        return date(int(cleaned), 1, 1)

    # ── Layer 5: dateutil fallback (handles 50+ formats) ──
    try:
        return dateutil_parser.parse(
            date_str,
            default=datetime(2000, 1, 1)
        ).date()
    except Exception:
        return None


# =========================================================
# OVERLAP CORRECTION (critical fix)
# =========================================================
def merge_intervals(intervals: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """
    Merges overlapping date ranges to prevent double-counting.
    e.g. [(2020-01, 2022-06), (2021-01, 2021-12)] → [(2020-01, 2022-06)]
    """
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            # Overlapping — extend if needed
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged

def intervals_to_months(intervals: list[tuple[date, date]]) -> float:
    merged = merge_intervals(intervals)
    total_days = sum((e - s).days for s, e in merged)
    return round(total_days / 30.44, 1)


# =========================================================
# SECTION EXTRACTOR
# =========================================================
EXPERIENCE_HEADERS = [
    "experience", "work experience", "professional experience",
    "employment history", "work history", "career history",
    "professional background", "internship"
]

STOP_HEADERS = [
    "education", "projects", "skills", "certifications",
    "summary", "objective", "awards", "publications",
    "languages", "interests", "hobbies", "references"
]

def extract_experience_section(text: str) -> str:
    """
    Extracts the work experience section using fuzzy header matching.
    Fixed: uses token_set_ratio instead of partial_ratio for better
    section header detection.
    """
    lines = text.split("\n")
    start_line = None
    end_line = None

    for i, line in enumerate(lines):
        clean = line.lower().strip()
        if not clean:
            continue
        for keyword in EXPERIENCE_HEADERS:
            # token_set_ratio handles word order + partial matches better
            if fuzz.token_set_ratio(clean, keyword) > 75:
                start_line = i
                break
        if start_line is not None:
            break

    if start_line is None:
        return text  # fallback: use full text

    for i in range(start_line + 1, len(lines)):
        clean = lines[i].lower().strip()
        if not clean:
            continue
        for keyword in STOP_HEADERS:
            if fuzz.token_set_ratio(clean, keyword) > 75:
                end_line = i
                break
        if end_line is not None:
            break

    end_line = end_line or len(lines)
    return "\n".join(lines[start_line:end_line])


# =========================================================
# JOB BLOCK SPLITTER
# =========================================================
def get_job_blocks(exp_text: str) -> list[str]:
    """Split experience section into individual job entries."""
    blocks = re.split(r'\n{2,}', exp_text)
    filtered = []

    for block in blocks:
        block = block.strip()
        if len(block) < 20:
            continue
        b_lower = block.lower()

        has_job_keyword = any(word in b_lower for word in JOB_KEYWORDS)
        has_date = any(
            re.search(p, block, re.IGNORECASE)
            for p in DATE_RANGE_PATTERNS
        )

        if has_job_keyword and has_date:
            filtered.append(block)

    return filtered


# =========================================================
# DATE RANGE EXTRACTOR FROM BLOCKS
# =========================================================
def extract_date_ranges(blocks: list[str]) -> list[tuple[date, date]]:
    ranges = []
    for block in blocks:
        for pattern in DATE_RANGE_PATTERNS:
            for match in re.finditer(pattern, block, re.IGNORECASE):
                start_str, end_str = match.groups()
                start = parse_date(start_str)
                end   = parse_date(end_str)
                if start and end and start <= end:
                    ranges.append((start, end))
    return ranges


# =========================================================
# ROLE-SPECIFIC EXPERIENCE
# =========================================================
def extract_role_intervals(blocks: list[str]) -> dict[str, list[tuple[date, date]]]:
    """
    For each job block, detect the role title and map to
    canonical role, then store its date interval.
    Returns {canonical_role: [(start, end), ...]}
    """
    role_intervals: dict[str, list] = {}

    for block in blocks:
        # Find date range in this block
        block_range = None
        for pattern in DATE_RANGE_PATTERNS:
            match = re.search(pattern, block, re.IGNORECASE)
            if match:
                start = parse_date(match.group(1))
                end   = parse_date(match.group(2))
                if start and end and start <= end:
                    block_range = (start, end)
                    break

        if not block_range:
            continue

        # Detect canonical role from block text
        canonical = get_canonical_role(block)
        if canonical:
            role_intervals.setdefault(canonical, [])
            role_intervals[canonical].append(block_range)

    return role_intervals


# =========================================================
# EXPERIENCE SCORE (non-linear, used by matcher.py)
# =========================================================
def experience_score(candidate_months: float, required_months: float) -> float:
    """
    Non-linear scoring:
    - Under-qualified: steep penalty (ratio^1.5)
    - Overqualified: capped at 1.0, no bonus
    """
    if required_months <= 0:
        return 1.0
    ratio = candidate_months / required_months
    if ratio >= 1.0:
        return min(1.0, 0.9 + 0.1 * math.log(ratio + 1))
    return round(ratio ** 1.5, 3)


# =========================================================
# MASTER FUNCTION
# =========================================================
@dataclass
class ExperienceResult:
    total_months: float
    total_years: float
    role_experience: dict[str, float]   # {canonical_role: months}
    date_ranges: list                    # raw for debugging
    blocks_found: int

def extract_work_experience(text: str) -> ExperienceResult:
    """
    Full experience extraction pipeline:
    1. Find experience section
    2. Split into job blocks
    3. Extract date ranges
    4. Merge overlapping intervals (NEW)
    5. Calculate role-specific experience (NEW)
    """
    exp_section = extract_experience_section(text)
    blocks      = get_job_blocks(exp_section)
    date_ranges = extract_date_ranges(blocks)

    # Total experience with overlap correction
    total_months = intervals_to_months(date_ranges)
    total_years  = round(total_months / 12, 2)

    # Role-specific breakdown
    role_intervals  = extract_role_intervals(blocks)
    role_experience = {
        role: intervals_to_months(intervals)
        for role, intervals in role_intervals.items()
    }

    return ExperienceResult(
        total_months    = total_months,
        total_years     = total_years,
        role_experience = role_experience,
        date_ranges     = date_ranges,
        blocks_found    = len(blocks),
    )


# =========================================================
# STANDALONE TEST
# =========================================================
if __name__ == "__main__":
    from pdf_parser import extract_text_from_pdf

    pdf_path = r"C:\Aqib_project\AI_powered_Resume_screening\resumes\Umer_Aftab_Resume.pdf"

    parsed = extract_text_from_pdf(pdf_path)   # now returns ParsedDocument
    result = extract_work_experience(parsed.text)

    print(f"Total Experience : {result.total_years} years ({result.total_months} months)")
    print(f"Job Blocks Found : {result.blocks_found}")
    print(f"Date Ranges      : {result.date_ranges}")
    print(f"Role Breakdown   : {result.role_experience}")