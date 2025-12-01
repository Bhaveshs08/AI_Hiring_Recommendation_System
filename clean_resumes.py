import os
import re
import json
import uuid
import glob

RAW_RESUME_DIR = "data/resumes"
CLEAN_RESUME_DIR = "data/resumes_cleaned"

os.makedirs(CLEAN_RESUME_DIR, exist_ok=True)

def extract_email(text):
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else f"resume_{uuid.uuid4()}@unknown.com"

def extract_name(text):
    # crude heuristic: first non-empty line that’s not email/phone
    for line in text.splitlines():
        if line.strip() and "@" not in line and not re.search(r"\d", line):
            return line.strip()
    return "Unknown Candidate"

def extract_skills(text):
    skills_section = re.findall(r"Skills[:\-]?(.*)", text, re.IGNORECASE)
    if skills_section:
        skills_line = skills_section[0]
        return [s.strip() for s in re.split(r"[,;]", skills_line) if s.strip()]
    return []

def process_resume(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    email = extract_email(text)
    name = extract_name(text)
    skills = extract_skills(text)

    cleaned = {
        "id": email,
        "email": email,
        "name": name,
        "professional_summary": text[:500],   # crude summary
        "experience_summary": text[:1000],    # crude experience block
        "raw_skills": skills,
        "location": "Unknown",
        "education": [],
        "certifications": [],
        "plaintext": text
    }

    return cleaned

def main():
    files = glob.glob(os.path.join(RAW_RESUME_DIR, "*.txt"))
    for file_path in files:
        cleaned = process_resume(file_path)
        out_file = os.path.join(
            CLEAN_RESUME_DIR,
            os.path.splitext(os.path.basename(file_path))[0] + ".json"
        )
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2)
        print(f"✅ Converted {file_path} → {out_file}")

if __name__ == "__main__":
    main()
