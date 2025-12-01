import os
import json
from pathlib import Path
from docx import Document
from PyPDF2 import PdfReader

# ==============================
# 📂 Directories
# ==============================
BASE_DIR = Path("data")
RAW_RESUMES = BASE_DIR / "resumes"
RAW_JDS = BASE_DIR / "job_descriptions"
CLEAN_RESUMES = BASE_DIR / "resumes_cleaned"
CLEAN_JDS = BASE_DIR / "job_descriptions_cleaned"

CLEAN_RESUMES.mkdir(parents=True, exist_ok=True)
CLEAN_JDS.mkdir(parents=True, exist_ok=True)


# ==============================
# 🔧 Helper Functions
# ==============================
def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text.append(extracted)
    return "\n".join(text)


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), indent=2)


def convert_to_json(input_path, output_dir, prefix):
    ext = input_path.suffix.lower()
    try:
        if ext == ".txt":
            text = read_txt(input_path)
        elif ext == ".docx":
            text = read_docx(input_path)
        elif ext == ".pdf":
            text = read_pdf(input_path)
        elif ext == ".json":
            text = read_json(input_path)
        else:
            print(f"⚠️ Skipping unsupported file: {input_path}")
            return

        output_data = {
            "id": input_path.stem,
            "source_file": input_path.name,
            "text": text.strip()
        }

        output_file = output_dir / f"{input_path.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Converted {input_path.name} → {output_file.name}")

    except Exception as e:
        print(f"❌ Error processing {input_path.name}: {e}")


def process_folder(input_dir, output_dir, prefix):
    for file in sorted(input_dir.glob("*")):
        convert_to_json(file, output_dir, prefix)


# ==============================
# 🚀 Main
# ==============================
def main():
    print("🔄 Converting resumes...")
    process_folder(RAW_RESUMES, CLEAN_RESUMES, "resume")

    print("\n🔄 Converting job descriptions...")
    process_folder(RAW_JDS, CLEAN_JDS, "jd")

    print("\n✅ All files converted successfully!")


if __name__ == "__main__":
    main()
def normalize_to_json(input_path, output_path):
    """
    Normalize a resume or JD into JSON format.
    Currently just wraps the conversion logic already used in file_converter.py.
    """
    import os, json

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Minimal normalization
    data = {
        "filename": os.path.basename(input_path),
        "content": text.strip()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path
