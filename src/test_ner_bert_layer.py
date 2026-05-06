"""
phase1_ner_test.py
Phase 1 validation: test the 4-layer BERT NER pipeline against real resumes.

Usage:
    # Without BERT (baseline — spaCy + heuristics only):
    python phase1_ner_test.py

    # With BERT as Layer 0 (set token first):
    $env:HF_API_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
    python phase1_ner_test.py
"""

import os
import sys
import time

# ── Make src imports work when running from src/ ──────────────────
sys.path.insert(0, os.path.dirname(__file__))

from pdf_parser import extract_text_from_pdf
from ner_extractor import (
    extract_name_bert,
    extract_name_spacy,
    extract_name_first_line,
    extract_name_regex,
    extract_name_ner,
    _HF_TOKEN,
)

# ── Test resumes — mix of categories ─────────────────────────────
TEST_PDFS = [
    (r"C:\Aqib_project\AI_powered_Resume_screening\resumes\data\ACCOUNTANT\10554236.pdf",  "ACCOUNTANT"),
    (r"C:\Aqib_project\AI_powered_Resume_screening\resumes\data\ACCOUNTANT\10674770.pdf",  "ACCOUNTANT"),
    (r"C:\Aqib_project\AI_powered_Resume_screening\resumes\data\ACCOUNTANT\11163645.pdf",  "ACCOUNTANT"),
    (r"C:\Aqib_project\AI_powered_Resume_screening\resumes\data\ENGINEERING\10030015.pdf", "ENGINEERING"),
    (r"C:\Aqib_project\AI_powered_Resume_screening\resumes\data\ENGINEERING\10219099.pdf", "ENGINEERING"),
]

LAYER_LABELS = {
    "bert_ner":       "Layer 0 — BERT (dslim/bert-base-NER)",
    "spacy_ner":      "Layer 1 — spaCy (en_core_web_sm)",
    "first_line":     "Layer 2 — First-line heuristic",
    "regex_fallback": "Layer 3 — Regex fallback",
    "failed":         "FAILED — no layer matched",
}

CONFIDENCE_BAR = {
    0.95: "█████ 0.95",
    0.90: "████░ 0.90",
    0.75: "███░░ 0.75",
    0.60: "██░░░ 0.60",
    0.00: "░░░░░ 0.00",
}


def confidence_bar(score: float) -> str:
    return CONFIDENCE_BAR.get(score, f"{'█' * int(score*5)}{'░' * (5-int(score*5))} {score:.2f}")


def per_layer_breakdown(full_text: str) -> dict:
    """Run each layer individually to show what each one would have extracted."""
    header = "\n".join(full_text.split("\n")[:30])
    return {
        "bert":       extract_name_bert(full_text),   # full text — uses sliding window
        "spacy":      extract_name_spacy(header),
        "first_line": extract_name_first_line(header),
        "regex":      extract_name_regex(header),
    }


def run_tests():
    print("\n" + "=" * 65)
    print("  PHASE 1 — NER LAYER VALIDATION")
    print(f"  BERT Token : {'SET [OK]' if _HF_TOKEN else 'NOT SET [--]  (BERT layer will be skipped)'}")
    print("=" * 65)

    layer_counts = {"bert_ner": 0, "spacy_ner": 0, "first_line": 0,
                    "regex_fallback": 0, "failed": 0}
    results = []

    for pdf_path, category in TEST_PDFS:
        filename = os.path.basename(pdf_path)
        print(f"\n{'─'*65}")
        print(f"  File     : {filename}  [{category}]")

        # Step 1: Parse PDF
        try:
            parsed = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"  ✗ PDF parse failed: {e}")
            continue

        header_text = "\n".join(parsed.text.split("\n")[:30])

        print(f"  Parse    : {parsed.extraction_method} (confidence {parsed.confidence:.2f})")
        print(f"  Header   :\n    {chr(10).join('    ' + l for l in header_text.split(chr(10))[:6])}")

        # Step 2: Run full 4-layer extraction
        t0 = time.time()
        result = extract_name_ner(parsed.text)
        elapsed = (time.time() - t0) * 1000

        # Step 3: Per-layer breakdown (shows what each layer sees)
        breakdown = per_layer_breakdown(parsed.text)

        print(f"\n  ── Final result ─────────────────────────────")
        print(f"  Name     : {result.value or '(none)'}")
        print(f"  Source   : {LAYER_LABELS.get(result.source, result.source)}")
        print(f"  Confidence: {confidence_bar(result.confidence)}")
        print(f"  Time     : {elapsed:.0f}ms")

        print(f"\n  ── Per-layer breakdown ──────────────────────")
        print(f"  BERT       → {breakdown['bert']       or '(no match)'}")
        print(f"  spaCy      → {breakdown['spacy']      or '(no match)'}")
        print(f"  First-line → {breakdown['first_line'] or '(no match)'}")
        print(f"  Regex      → {breakdown['regex']      or '(no match)'}")

        layer_counts[result.source] = layer_counts.get(result.source, 0) + 1
        results.append({
            "file": filename,
            "category": category,
            "name": result.value,
            "source": result.source,
            "confidence": result.confidence,
        })

    # ── Summary ───────────────────────────────────────────────────
    total = len(results)
    found = sum(1 for r in results if r["name"])

    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Resumes tested   : {total}")
    print(f"  Names found      : {found} / {total}  ({found/total*100:.0f}%)" if total else "  No results")
    print(f"\n  Layer firing counts:")
    for layer, count in layer_counts.items():
        if count:
            bar = "█" * count + "░" * (total - count)
            print(f"    {LAYER_LABELS[layer]:<42} {bar}  {count}")
    print(f"\n  {'BERT ACTIVE [OK]' if _HF_TOKEN else 'BERT NOT ACTIVE -- set HF_API_TOKEN to enable Layer 0'}")
    print("=" * 65 + "\n")

    if not _HF_TOKEN:
        print("  To activate BERT (Layer 0):")
        print("    PowerShell : $env:HF_API_TOKEN = 'hf_xxxxxxxxxxxxxxxxxxxx'")
        print("    Then re-run: python phase1_ner_test.py\n")


if __name__ == "__main__":
    run_tests()
