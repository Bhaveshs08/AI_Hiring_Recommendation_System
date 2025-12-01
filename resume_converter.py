import os
import json

RAW_DIR = "data/resumes"
CLEANED_DIR = "data/resumes_cleaned"

# Ensure cleaned dir exists
os.makedirs(CLEANED_DIR, exist_ok=True)

def clean_resume(resume):
    return {
        "id": resume.get("id", ""),
        "name": resume.get("name", ""),
        "location": resume.get("contact", {}).get("location", ""),
        "skills": resume.get("technical_skills", []),
        "experience": resume.get("experience", []),
        "projects": resume.get("projects", []),
    }

for file in os.listdir(RAW_DIR):
    if file.endswith(".json"):
        with open(os.path.join(RAW_DIR, file), "r", encoding="utf-8") as f:
            raw_resume = json.load(f)

        cleaned = clean_resume(raw_resume)

        with open(os.path.join(CLEANED_DIR, file), "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2)

print(f"Resumes cleaned and saved to {CLEANED_DIR}")
