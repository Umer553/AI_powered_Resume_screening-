"""
phase2_test.py
Validates Phase 2: multi-entity BERT extraction (PER + ORG + LOC).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pdf_parser import extract_text_from_pdf
from ner_extractor import extract_entities_bert, extract_all_contact_info
from information_extractor import extract_candidate_info

TEST_PDFS = [
    (r"C:\Aqib_project\AI_powered_Resume_screening\resumes\data\ACCOUNTANT\11163645.pdf", "ACCOUNTANT"),
    (r"C:\Aqib_project\AI_powered_Resume_screening\resumes\data\ACCOUNTANT\10554236.pdf", "ACCOUNTANT"),
    (r"C:\Aqib_project\AI_powered_Resume_screening\resumes\data\ENGINEERING\10030015.pdf", "ENGINEERING"),
]

from ner_extractor import _HF_TOKEN
print("=" * 60)
print("  PHASE 2 -- Multi-Entity BERT Extraction (PER + ORG + LOC)")
print(f"  BERT Token : {'SET [OK]' if _HF_TOKEN else 'NOT SET'}")
print("=" * 60)

for pdf_path, category in TEST_PDFS:
    fname = os.path.basename(pdf_path)
    print(f"\n--- {fname}  [{category}] ---")
    try:
        parsed = extract_text_from_pdf(pdf_path)
    except Exception as e:
        print(f"  Parse failed: {e}")
        continue

    # Raw multi-entity BERT call
    bert = extract_entities_bert(parsed.text)
    print(f"  BERT PER (name)    : {bert['per'] or '(none)'}")
    print(f"  BERT ORG (employers): {bert['orgs'] or '(none)'}")
    print(f"  BERT LOC (location): {bert['location'] or '(none)'}")

    # Full contact info pipeline
    contact = extract_all_contact_info(parsed.text)
    print(f"\n  extract_all_contact_info:")
    print(f"    name      : {contact['name']}  [{contact['name_source']} @ {contact['name_confidence']}]")
    print(f"    email     : {contact['email']}")
    print(f"    phone     : {contact['phone']}")
    print(f"    location  : {contact['location']}")
    print(f"    companies : {contact['companies']}")

    # Full candidate info
    info = extract_candidate_info(parsed.text, parsed.confidence)
    print(f"\n  extract_candidate_info (new fields):")
    print(f"    location  : {info['location']}")
    print(f"    companies : {info['companies']}")

print("\n" + "=" * 60)
print("  Phase 2 test complete.")
print("=" * 60)
