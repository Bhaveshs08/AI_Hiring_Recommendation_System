#!/usr/bin/env python3
"""
list_jd_ids.py

List all vector IDs present in a Pinecone namespace (job descriptions).
Works with Pinecone SDK v4+ shapes (generator pages, dict pages, lists).
Outputs:
 - printed list of ids (one-per-line)
 - summary counts
 - option to save to CSV/JSON

Usage:
  python list_jd_ids.py --namespace Job_Descriptions --index polaris
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Pinecone client
from pinecone import Pinecone

# ---------------- args ----------------
parser = argparse.ArgumentParser(description="List vector ids from a Pinecone namespace.")
parser.add_argument("--namespace", "-n", required=True, help="Pinecone namespace to list (e.g. Job_Descriptions)")
parser.add_argument("--index", "-i", required=False, help="Pinecone index name (or set PINECONE_INDEX env var)")
parser.add_argument("--api-key", required=False, help="Pinecone API key (or set PINECONE_API_KEY env var)")
parser.add_argument("--limit", type=int, default=100, help="Page limit for each list() call (must be >0 and <=100). Default=100")
parser.add_argument("--out-csv", required=False, help="Optional path to save ids as CSV")
parser.add_argument("--out-json", required=False, help="Optional path to save ids as JSON")
args = parser.parse_args()

NAMESPACE = args.namespace
INDEX_NAME = args.index or os.environ.get("PINECONE_INDEX")
PINECONE_KEY = args.api_key or os.environ.get("PINECONE_API_KEY")
LIST_LIMIT = max(1, min(100, int(args.limit)))  # clamp to 1..100

if not PINECONE_KEY or not INDEX_NAME:
    print("ERROR: set PINECONE_API_KEY and PINECONE_INDEX (or pass --api-key/--index).", file=sys.stderr)
    sys.exit(2)

# ---------------- connect ----------------
pc = Pinecone(api_key=PINECONE_KEY)
ix = pc.Index(INDEX_NAME)

print(f"Fetching index list (page limit={LIST_LIMIT}) from namespace: {NAMESPACE}")

all_ids = []
try:
    pages = ix.list(namespace=NAMESPACE, limit=LIST_LIMIT)
except Exception as e:
    print("ERROR: Index.list() call failed:", e, file=sys.stderr)
    sys.exit(1)

# pages may be a generator -> iterate
try:
    for page in pages:
        # page might be:
        # - a list of ids (old shape)
        # - a dict with keys like 'ids', 'matches', or id->meta map
        # - a generator yielding strings (rare)
        if page is None:
            continue

        # If page is a string: treat as single id (rare)
        if isinstance(page, str):
            all_ids.append(page)
            continue

        # If page is list of strings or list of dicts
        if isinstance(page, list):
            # list of ids (strings)
            if page and all(isinstance(x, str) for x in page):
                all_ids.extend(page)
                continue
            # list of dicts (e.g., matches)
            for entry in page:
                if isinstance(entry, dict):
                    if "id" in entry:
                        all_ids.append(entry["id"])
                    elif "value" in entry and isinstance(entry["value"], dict) and "id" in entry["value"]:
                        all_ids.append(entry["value"]["id"])
            continue

        # If page is dict
        if isinstance(page, dict):
            # new SDK may return {'ids': [...]} or {'matches': [{'id':...}, ...]}
            if "ids" in page and isinstance(page["ids"], list):
                all_ids.extend([i for i in page["ids"] if isinstance(i, str)])
                continue
            if "matches" in page and isinstance(page["matches"], list):
                for m in page["matches"]:
                    if isinstance(m, dict) and "id" in m:
                        all_ids.append(m["id"])
                continue
            # older SDK shape: dict of id -> metadata
            # collect keys that look like ids (strings)
            try:
                keys = list(page.keys())
                if keys and all(isinstance(k, str) for k in keys):
                    # Heuristic: keys are likely ids
                    all_ids.extend(keys)
                    continue
            except Exception:
                pass

        # fallback: try to json-dump and search for "id" strings (best-effort)
        try:
            j = json.dumps(page)
            # naive extraction of "id":"<value>"
            # not perfect, but helps in odd shapes
            import re
            found = re.findall(r'"id"\s*:\s*"([^"]+)"', j)
            if found:
                all_ids.extend(found)
        except Exception:
            pass

except Exception as e:
    print("Error while iterating pages:", e, file=sys.stderr)
    sys.exit(1)

# deduplicate and sort (stable)
unique_ids = []
seen = set()
for _id in all_ids:
    if _id and _id not in seen:
        unique_ids.append(_id)
        seen.add(_id)

# ---------------- output ----------------
count = len(unique_ids)
print(f"Total ids scanned: {len(all_ids)}  unique ids found: {count}")

# show sample by grouping by candidate prefix if applicable (nice)
# Print up to 500 ids
display_n = min(500, count)
for i, vid in enumerate(unique_ids[:display_n], start=1):
    print(vid)

# save optional outputs
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.out_csv:
    import csv
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id"])
        for _id in unique_ids:
            writer.writerow([_id])
    print(f"Saved {len(unique_ids)} ids to CSV: {outp}")

if args.out_json:
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump({"namespace": NAMESPACE, "index": INDEX_NAME, "count": count, "ids": unique_ids}, fh, indent=2, ensure_ascii=False)
    print(f"Saved {len(unique_ids)} ids to JSON: {outp}")

# small grouping summary (by prefix until first '_' or full id)
from collections import Counter
groups = [ (i.split("_",1)[0] if "_" in i else i) for i in unique_ids ]
cnt = Counter(groups)
print("\nNamespaces / candidate groups summary (top 20):")
for k,v in cnt.most_common(20):
    print(f"  {k:30} count={v}")

print("\nDone.")
