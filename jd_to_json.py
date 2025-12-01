#!/usr/bin/env python3
"""
jd_to_json.py
Convert raw JD text/docx in data/jds_raw -> standardized JSON in data/jds/
Each output is a single JSON object (not a list); keys: jd_id, title, primary_skills, experience_required, description, metadata
"""
import os, sys, json, argparse, uuid, re
from pathlib import Path
from datetime import datetime
try:
    import docx
except Exception:
    docx = None

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")

def read_docx(path: Path) -> str:
    if not docx:
        raise RuntimeError("python-docx required to read .docx (pip install python-docx)")
    d = docx.Document(str(path))
    return "\n\n".join(p.text for p in d.paragraphs if p.text and p.text.strip())

def simple_extract_skills(text: str):
    # naive: look for "Skills:" or "Primary Skills" line and take following comma list
    m = re.search(r"(skills[:\s]*)(.+)", text, flags=re.IGNORECASE)
    if m:
        line = m.group(2).splitlines()[0]
        parts = [x.strip() for x in re.split(r"[,\|;/]", line) if x.strip()]
        return parts
    # fallback: return empty
    return []

def normalize_id(s: str):
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s[:48] or f"jd_{uuid.uuid4().hex[:6]}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/jds_raw", help="Source folder for raw JDs")
    parser.add_argument("--out", default="data/jds", help="Output folder for JSON JDs")
    args = parser.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in src.iterdir() if p.suffix.lower() in (".txt", ".docx")])
    if not files:
        print("No JD files found in", src)
        return
    for p in files:
        try:
            raw = read_docx(p) if p.suffix.lower() == ".docx" else read_txt(p)
            # title heuristic: first non-empty line
            title = next((l.strip() for l in raw.splitlines() if l.strip()), p.stem)
            skills = simple_extract_skills(raw)
            exp_m = re.search(r"(\d+\s*-\s*\d+\s*years|\d+\+\s*years|\d+\s*years)", raw, flags=re.IGNORECASE)
            exp = exp_m.group(0) if exp_m else ""
            jd_id = f"jd_{normalize_id(p.stem)}"
            obj = {
                "jd_id": jd_id,
                "title": title,
                "primary_skills": skills,
                "experience_required": exp,
                "description": raw[:30000],
                "metadata": {
                    "source_file": p.name,
                    "created_at": datetime.now().isoformat()
                }
            }
            out_file = out / (jd_id + ".json")
            with open(out_file, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, ensure_ascii=False, indent=2)
            print("Saved JD:", out_file.name)
        except Exception as e:
            print("ERROR processing", p, ":", e, file=sys.stderr)

if __name__ == "__main__":
    main()
