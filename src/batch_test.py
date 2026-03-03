"""
batch_test.py — Entry Point
AI-Powered Resume Screening System

Usage:
    python batch_test.py
    python batch_test.py --folder "path/to/resumes"
    python batch_test.py --jd "path/to/jd.txt"
"""

import os
import sys
import argparse
from ranker import rank_resumes

# =========================================================
# DEFAULT PATHS
# =========================================================
DEFAULT_RESUME_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "..", "resumes", "data"
)

# =========================================================
# JD INPUT — runtime input, not hardcoded
# =========================================================
SAMPLE_JD = """
We are hiring a Machine Learning Engineer.

Requirements:
- Strong Python programming skills
- PyTorch or TensorFlow experience
- Natural Language Processing (NLP)
- Deep Learning and neural networks
- Data preprocessing and feature engineering
- Model deployment and MLOps
- REST APIs using FastAPI or Flask
- Docker and version control (Git)

Minimum 2+ years of relevant experience required.
"""

def get_job_description(jd_file: str = None) -> str:
    """
    Gets JD from:
    1. File path argument (--jd flag)
    2. Interactive input (user pastes at runtime)
    3. Sample JD fallback (for testing)
    """
    # From file
    if jd_file and os.path.exists(jd_file):
        with open(jd_file, "r", encoding="utf-8") as f:
            jd = f.read().strip()
        print(f"✅ JD loaded from file: {jd_file}")
        return jd

    # Interactive input
    print("\n" + "="*55)
    print("  PASTE JOB DESCRIPTION")
    print("  (paste text, then press Enter twice to finish)")
    print("="*55)

    lines = []
    try:
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        jd = "\n".join(lines).strip()
    except EOFError:
        jd = ""

    # Fallback to sample if nothing entered
    if not jd:
        print("\n⚠ No JD entered — using sample ML Engineer JD for testing\n")
        jd = SAMPLE_JD

    return jd


# =========================================================
# ARGUMENT PARSER
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Powered Resume Screening System"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=DEFAULT_RESUME_FOLDER,
        help="Path to resume folder (searches subfolders automatically)"
    )
    parser.add_argument(
        "--jd",
        type=str,
        default=None,
        help="Path to job description .txt file (optional)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run with sample JD against all resumes (quick test mode)"
    )
    return parser.parse_args()


# =========================================================
# ENTRY POINT
# =========================================================
def main():
    args = parse_args()

    # ── Get job description ───────────────────────────────
    if args.test:
        print("\n🧪 TEST MODE — using sample ML Engineer JD")
        jd_text = SAMPLE_JD
    else:
        jd_text = get_job_description(args.jd)

    # ── Validate resume folder ────────────────────────────
    resume_folder = args.folder
    if not os.path.exists(resume_folder):
        print(f"\n❌ Resume folder not found: {resume_folder}")
        print("   Update DEFAULT_RESUME_FOLDER or use --folder flag")
        sys.exit(1)

    # ── Run full pipeline via ranker.py ───────────────────
    # Everything else (parsing, extraction, scoring,
    # ranking, export) is handled inside rank_resumes()
    results = rank_resumes(
        job_description = jd_text,
        resume_folder   = resume_folder,
    )

    if not results:
        print("\n⚠ No results generated. Check resume folder and PDF quality.")
        return

    # ── Quick top-3 summary ───────────────────────────────
    print("\n🏆 TOP CANDIDATES:\n")
    for r in results[:3]:
        print(f"  #{r.rank} {r.name}")
        print(f"     Score    : {r.final_score*100:.1f}%")
        print(f"     Decision : {r.decision}")
        print(f"     Skills   : {', '.join(r.matched_skills[:5])}")
        print(f"     Missing  : {', '.join(r.missing_skills[:3])}")
        print()

    print(f"📁 Full results saved to: outputs/")
    print(f"   Run again with a different JD to compare rankings.\n")


if __name__ == "__main__":
    main()
