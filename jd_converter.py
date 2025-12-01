import os
import json

RAW_JD_DIR = "data/job_descriptions"
CLEANED_JD_DIR = "data/job_descriptions_cleaned"

os.makedirs(CLEANED_JD_DIR, exist_ok=True)

def clean_job_description(file_path):
    """Load and clean a job description JSON into standard format."""
    with open(file_path, "r", encoding="utf-8") as f:
        jd = json.load(f)

    cleaned = {
        "id": jd.get("id", os.path.basename(file_path).split(".")[0]),
        "title": jd.get("title", ""),
        "company": jd.get("company", ""),
        "location": jd.get("location", ""),
        "requirements": jd.get("requirements", []),
        "responsibilities": jd.get("responsibilities", []),
        "skills_required": jd.get("skills_required", []),
        "experience_required": jd.get("experience_required", ""),
        "description": jd.get("description", "")
    }
    return cleaned

def process_all_jds():
    for file_name in os.listdir(RAW_JD_DIR):
        if file_name.endswith(".json"):
            file_path = os.path.join(RAW_JD_DIR, file_name)
            cleaned = clean_job_description(file_path)

            save_path = os.path.join(CLEANED_JD_DIR, file_name)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, indent=2, ensure_ascii=False)

            print(f"✅ Cleaned JD saved: {save_path}")

if __name__ == "__main__":
    process_all_jds()
