#!/usr/bin/env python3
# jd_verify.py
"""
Quick JD verification script.
Lists IDs in Job_Descriptions namespace (page size 99) and prints metadata summary for a sample.
"""

import os, sys
from pinecone import Pinecone
from pathlib import Path
import json

# ensure logs dir
Path("logs").mkdir(exist_ok=True)

# UTF-8 safe stdout
try:
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX")
NAMESPACE = "Job_Descriptions"
SAMPLE_N = 5
LIST_LIMIT = 99   # must be <100 per Pinecone error seen earlier

if not PINECONE_KEY or not INDEX_NAME:
    print("ERROR: set PINECONE_API_KEY and PINECONE_INDEX environment variables", file=sys.stderr)
    sys.exit(2)

pc = Pinecone(api_key=PINECONE_KEY)
ix = pc.Index(INDEX_NAME)

def list_ids(namespace, limit=LIST_LIMIT):
    all_ids = []
    try:
        for page in ix.list(namespace=namespace, limit=limit):
            # page can be list/str/dict depending on SDK shape
            if isinstance(page, str):
                all_ids.append(page)
            elif isinstance(page, list):
                all_ids.extend([p for p in page if isinstance(p, str)])
            elif isinstance(page, dict):
                # older SDK shape: dict => keys are ids
                all_ids.extend(list(page.keys()))
    except Exception as e:
        print("Error listing index ids:", e, file=sys.stderr)
        return []
    return [i for i in all_ids if i]

def fetch_and_print(sample_ids):
    if not sample_ids:
        print("No JD ids found in namespace.")
        return
    try:
        fetched = ix.fetch(ids=sample_ids, namespace=NAMESPACE)
    except Exception as e:
        print("Fetch failed:", e, file=sys.stderr)
        return
    vecs = getattr(fetched, "vectors", None) or (fetched.get("vectors") if isinstance(fetched, dict) else {}) or {}
    print(f"Verifying {len(sample_ids)} JD vectors (namespace={NAMESPACE}):")
    print("-" * 72)
    for vid in sample_ids:
        v = vecs.get(vid) if isinstance(vecs, dict) else None
        print(f"jd_id: {vid}")
        if not v:
            print("  WARNING: no vector returned for this id.")
            print("-" * 72)
            continue
        meta = getattr(v, "metadata", None) or (v.get("metadata") if isinstance(v, dict) else {}) or {}
        title = meta.get("title") or meta.get("job_title") or ""
        exp = meta.get("experience_required") or meta.get("experience") or ""
        skills = meta.get("primary_skills") or meta.get("skills") or []
        src = meta.get("source_file") or meta.get("source") or None
        snippet = (meta.get("pagecontent") or meta.get("description") or "")[:400]
        print(f"  title: {title}")
        print(f"  experience_required: {exp}")
        print(f"  primary_skills: {', '.join(skills) if isinstance(skills, (list,tuple)) else skills}")
        print(f"  source_file: {src}")
        print(f"  snippet: {snippet.replace(chr(10),' ') if snippet else '(not available in metadata)'}")
        print("-" * 72)

def main():
    ids = list_ids(NAMESPACE)
    if not ids:
        print("Verifying 0 JD vectors (namespace=" + NAMESPACE + "):")
        print("-" * 72)
        print("No JD ids found in namespace.")
        return
    sample = ids[:SAMPLE_N]
    fetch_and_print(sample)

if __name__ == "__main__":
    main()
