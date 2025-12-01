#!/usr/bin/env python3
"""
txt_to_json.py
Convert plain-text / docx resumes in data/resumes_raw -> standardized JSON files in data/resumes/.
Each output is a JSON file (list of chunk objects). Candidate ID is UUID by default.
Optional: call OpenAI (Responses) to create section summaries (requires OPENAI_API_KEY).
"""
import os, sys, json, argparse, uuid, re
from pathlib import Path
from datetime import datetime

try:
    import docx   # python-docx
except Exception:
    docx = None

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")

def read_docx(path: Path) -> str:
    if not docx:
        raise RuntimeError("python-docx not installed (pip install python-docx) to read .docx files")
    d = docx.Document(str(path))
    return "\n\n".join(p.text for p in d.paragraphs if p.text and p.text.strip())

def split_into_sections(text: str) -> list:
    # naive heuristic: headings lines all-caps or common section words
    lines = [l.rstrip() for l in text.splitlines()]
    sections = []
    cur_lines = []
    cur_title = None
    for ln in lines:
        # treat heading if it's short and contains keywords or is uppercase
        if ln.strip() and (ln.strip().upper() == ln.strip() and len(ln.strip()) < 60) and len(ln.strip().split()) < 6:
            # push previous
            if cur_lines:
                sections.append((cur_title or "BODY", "\n".join(cur_lines).strip()))
                cur_lines = []
            cur_title = ln.strip()
        else:
            if ln is not None:
                cur_lines.append(ln)
    if cur_lines:
        sections.append((cur_title or "BODY", "\n".join(cur_lines).strip()))
    return sections if sections else [("BODY", text.strip())]

def extract_name_email(text: str):
    # very simple heuristics
    name = ""
    email = None
    # first non-empty line maybe name
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # if line looks like a name (few words, letters)
        if 1 < len(s.split()) <= 4 and any(c.isalpha() for c in s):
            name = s
            break
    m = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
    if m:
        email = m.group(1).lower()
    return name, email

def normalize_filename(name: str):
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)

def make_chunk_objects(file_path: Path, force_uuid=True):
    raw = read_docx(file_path) if file_path.suffix.lower() == ".docx" else read_txt(file_path)
    name, email = extract_name_email(raw)
    candidate_id = f"resumes_{uuid.uuid4().hex[:12]}" if force_uuid else f"resumes_{normalize_filename(name or file_path.stem)}_{uuid.uuid4().hex[:6]}"
    sections = split_into_sections(raw)
    chunks = []
    for idx, (sec_title, sec_text) in enumerate(sections, start=1):
        meta = {
            "candidate_id": candidate_id,
            "candidate_name": name or "",
            "email": email or "",
            "section": sec_title or f"section{idx}",
            "source_file": file_path.name,
            "version": 1,
            "updated_at": datetime.now().isoformat()
        }
        chunks.append({
            "chunk_index": idx,
            "section": sec_title,
            "chunk_text": sec_text,
            "metadata": meta
        })
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/resumes_raw", help="Source folder with .txt/.docx resumes")
    parser.add_argument("--out", default="data/resumes", help="Output folder for JSON resumes")
    parser.add_argument("--force-uuid", action="store_true", help="Always use UUID for candidate_id (recommended)")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.iterdir() if p.suffix.lower() in (".txt", ".docx")])
    if not files:
        print("No .txt or .docx resume files found in", src)
        return

    for p in files:
        try:
            chunks = make_chunk_objects(p, force_uuid=args.force_uuid)
            out_file = out / (p.stem + ".json")
            # ensure valid json list
            with open(out_file, "w", encoding="utf-8") as fh:
                json.dump(chunks, fh, ensure_ascii=False, indent=2)
            print("Converted:", p.name, "->", out_file.name, "chunks=", len(chunks))
        except Exception as e:
            print("ERROR converting", p, ":", e, file=sys.stderr)

if __name__ == "__main__":
    main()
