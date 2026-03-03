from pdf_parser import extract_text_from_pdf
# from ranker import calculate_similarity
from matcher import compute_match_score
from information_extractor import extract_candidate_info

resume_path = r"C:\Aqib_project\AI_powered_Resume_screening\Umer_Aftab_Resume.pdf"

text = extract_text_from_pdf(resume_path)

print("\n========== EXTRACTED TEXT ==========\n")
print(text)

info = extract_candidate_info(text)

for key, value in info.items():
    print(f"{key.upper()}: {value}\n")


print("\n========== MATCHING WITH JOB DESCRIPTION ==========\n")

job_description = """
We are looking for a Machine Learning Engineer with experience in deep learning,
NLP, Python, PyTorch or TensorFlow, model deployment, and data preprocessing.
Experience with APIs and backend integration is a plus.
"""

score = compute_match_score(text, job_description)

print("MATCH SCORE:", score, "%")
