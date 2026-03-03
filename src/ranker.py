"""
ranker.py
Batch processing pipeline — processes all PDFs, ranks candidates,
exports results. Called by batch_test.py and dashboard.py.
"""

import os
import json
import csv
from datetime import datetime
from dataclasses import dataclass

from pdf_parser            import extract_text_from_pdf
from information_extractor import extract_candidate_info
from experience_extractor  import extract_work_experience
from matcher               import compute_match_score, hiring_decision
from domain_config         import detect_domain


# =========================================================
# CANDIDATE RESULT — structured output for every resume
# =========================================================
@dataclass
class CandidateResult:
    # Identity
    rank:               int
    name:               str
    email:              str
    phone:              str
    linkedin:           str
    github:             str

    # Scores (all 0.0 - 1.0 internally)
    final_score:        float
    semantic_score:     float
    skill_score:        float
    total_exp_score:    float
    role_exp_score:     float

    # Experience
    total_exp_years:    float
    total_exp_months:   float
    role_exp_months:    float
    required_exp_years: float

    # Skills
    matched_skills:     list
    semantic_skills:    list
    missing_skills:     list
    skill_breakdown:    dict

    # Metadata
    domain:             str
    scoring_weights:    dict
    decision:           str
    file_path:          str
    parse_method:       str       # "pdfplumber" or "ocr"
    parse_confidence:   float
    needs_review:       bool


# =========================================================
# PDF COLLECTOR — recursive subfolder walk
# =========================================================
def collect_pdf_paths(resume_folder: str) -> list:
    """
    Finds all PDFs recursively.
    Handles: resumes/data/ML_ENGINEER/*.pdf
             resumes/data/TEACHER/*.pdf  etc.
    """
    pdf_paths = []
    for root, dirs, files in os.walk(resume_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths


# =========================================================
# SINGLE RESUME PROCESSOR
# Called by both rank_resumes() and dashboard.py directly
# =========================================================
def process_single_resume(
    pdf_path:      str,
    jd_text:       str,
    domain_name:   str,
    domain_config: dict,
) -> CandidateResult | None:
    """
    Full pipeline for one resume.
    Returns None on failure — batch continues uninterrupted.
    """
    try:
        # Step 1: Parse PDF
        parsed = extract_text_from_pdf(pdf_path)
        if not parsed.text or len(parsed.text.strip()) < 50:
            print(f"  ⚠ Skipping {os.path.basename(pdf_path)} — insufficient text")
            return None

        # Step 2: Extract candidate info
        info = extract_candidate_info(parsed.text, parsed.confidence)

        # Step 3: Extract experience
        exp_result = extract_work_experience(parsed.text)

        # Step 4: Compute match score
        match = compute_match_score(
            resume_text      = parsed.text,
            jd_text          = jd_text,
            candidate_skills = info["skills"],
            exp_result       = exp_result,
            domain_name      = domain_name,
            domain_config    = domain_config,
        )

        # Step 5: Hiring decision
        decision = hiring_decision(match["final_score"])

        return CandidateResult(
            rank               = 0,          # assigned after sorting
            name               = info.get("name")     or os.path.basename(pdf_path),
            email              = info.get("email")    or "N/A",
            phone              = info.get("phone")    or "N/A",
            linkedin           = info.get("linkedin") or "N/A",
            github             = info.get("github")   or "N/A",

            final_score        = match["final_score"],
            semantic_score     = match["semantic_score"],
            skill_score        = match["skill_score"],
            total_exp_score    = match["total_exp_score"],
            role_exp_score     = match["role_exp_score"],

            total_exp_years    = exp_result.total_years,
            total_exp_months   = exp_result.total_months,
            role_exp_months    = match["role_exp_months"],
            required_exp_years = match["required_exp_years"],

            matched_skills     = match["matched_skills"],
            semantic_skills    = match["semantic_skills"],
            missing_skills     = match["missing_skills"],
            skill_breakdown    = match["skill_breakdown"],

            domain             = match["domain"],
            scoring_weights    = match["scoring_weights_used"],
            decision           = decision,
            file_path          = pdf_path,
            parse_method       = parsed.extraction_method,
            parse_confidence   = parsed.confidence,
            needs_review       = info.get("needs_review", False),
        )

    except Exception as e:
        print(f"  ❌ Failed: {os.path.basename(pdf_path)} — {e}")
        return None


# =========================================================
# BATCH RANKER
# =========================================================
def rank_resumes(
    job_description: str,
    resume_folder:   str = None,
) -> list:
    """
    Full batch pipeline:
    1. Detect domain from JD (once per batch)
    2. Collect all PDFs recursively
    3. Process each resume with error isolation
    4. Sort by final score, assign ranks
    5. Export JSON + CSV to outputs/
    6. Print summary

    Returns list of CandidateResult dataclasses.
    """
    if resume_folder is None:
        resume_folder = os.path.join(
            os.path.dirname(__file__), "..", "resumes", "data"
        )

    # ── Step 1: Detect domain ─────────────────────────────
    print("\n" + "="*55)
    print("  AI-POWERED RESUME SCREENING SYSTEM")
    print("="*55)

    domain_name, domain_config = detect_domain(job_description)
    print(f"  Domain   : {domain_name}")
    print(f"  Weights  : {domain_config['scoring_weights']}")
    print("="*55 + "\n")

    # ── Step 2: Collect PDFs ──────────────────────────────
    pdf_paths = collect_pdf_paths(resume_folder)
    total     = len(pdf_paths)

    if total == 0:
        print(f"⚠ No PDFs found in: {resume_folder}")
        return []

    print(f"📂 Found {total} resumes — processing...\n")

    # ── Step 3: Process each resume ───────────────────────
    results = []
    for i, path in enumerate(pdf_paths, 1):
        print(f"[{i:>4}/{total}] {os.path.basename(path)}")
        result = process_single_resume(
            pdf_path      = path,
            jd_text       = job_description,
            domain_name   = domain_name,
            domain_config = domain_config,
        )
        if result:
            results.append(result)

    # ── Step 4: Sort + assign ranks ───────────────────────
    results.sort(key=lambda x: x.final_score, reverse=True)
    for i, candidate in enumerate(results, 1):
        candidate.rank = i

    # ── Step 5: Export ────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    csv_path  = os.path.join(output_dir, f"results_{timestamp}.csv")

    _export_json(results, json_path)
    _export_csv(results,  csv_path)

    # ── Step 6: Print summary ─────────────────────────────
    _print_summary(results, domain_name)

    return results


# =========================================================
# EXPORTERS
# =========================================================
def _export_json(results: list, path: str):
    output = []
    for r in results:
        d = r.__dict__.copy()
        output.append(d)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 JSON → {path}")


def _export_csv(results: list, path: str):
    if not results:
        return
    rows = []
    for r in results:
        rows.append({
            "Rank":             r.rank,
            "Name":             r.name,
            "Email":            r.email,
            "Phone":            r.phone,
            "Final Score %":    round(r.final_score     * 100, 1),
            "Semantic %":       round(r.semantic_score  * 100, 1),
            "Skill %":          round(r.skill_score     * 100, 1),
            "Exp Score %":      round(r.total_exp_score * 100, 1),
            "Role Exp Score %": round(r.role_exp_score  * 100, 1),
            "Total Exp Years":  r.total_exp_years,
            "Domain":           r.domain,
            "Decision":         r.decision,
            "Matched Skills":   ", ".join(r.matched_skills),
            "Missing Skills":   ", ".join(r.missing_skills),
            "LinkedIn":         r.linkedin,
            "GitHub":           r.github,
            "Parse Method":     r.parse_method,
            "Parse Confidence": round(r.parse_confidence * 100, 1),
            "Needs Review":     r.needs_review,
            "File":             os.path.basename(r.file_path),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"📊 CSV  → {path}")


# =========================================================
# SUMMARY PRINTER
# =========================================================
def _print_summary(results: list, domain: str):
    print("\n" + "="*55)
    print(f"  RESULTS — {domain.upper().replace('_', ' ')}")
    print("="*55)

    for r in results:
        flag = " ⚠" if r.needs_review else ""
        print(
            f"  #{r.rank:<4} {r.name:<28} "
            f"{r.final_score*100:>5.1f}%  {r.decision}{flag}"
        )

    print("="*55)

    hired  = sum(1 for r in results if "Hire" in r.decision
                 and "Maybe" not in r.decision)
    maybe  = sum(1 for r in results if "Maybe" in r.decision)
    review = sum(1 for r in results if r.needs_review)

    print(f"  Processed   : {len(results)}")
    print(f"  Recommended : {hired}")
    print(f"  Maybe       : {maybe}")
    print(f"  Review Flag : {review}")
    print("="*55 + "\n")