#!/usr/bin/env python3
# list_candidate_vectors.py
import os, sys
from pinecone import Pinecone
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX")
if not PINECONE_KEY or not INDEX_NAME:
    print("Set PINECONE_API_KEY and PINECONE_INDEX in env"); sys.exit(1)
pc = Pinecone(api_key=PINECONE_KEY)
ix = pc.Index(INDEX_NAME)
ns = "Resumes"
print("Fetching index list (first 10000 ids) from namespace:", ns)
# use list to get vector ids (SDK may paginate)
ids = []
for page in ix.list(namespace=ns, limit=100):
    # page may be list/str/dict depending on SDK; normalize
    if isinstance(page, str):
        ids.append(page)
    elif isinstance(page, list):
        ids.extend([p for p in page if isinstance(p, str)])
    elif isinstance(page, dict):
        if "matches" in page:
            ids.extend([m.get("id") for m in page.get("matches", []) if m.get("id")])
# fallback: unique
ids = [i for i in ids if i]
ids = list(dict.fromkeys(ids))
print(f"Total ids scanned: {len(ids)}")
# fetch batches of metadata
from math import ceil
def chunked(l,n): 
    for i in range(0,len(l),n):
        yield l[i:i+n]
mapping = {}
for batch in chunked(ids, 100):
    fetched = ix.fetch(ids=batch, namespace=ns)
    vecs = getattr(fetched, "vectors", None) or (fetched.get("vectors") if isinstance(fetched, dict) else {})
    for vid, v in (vecs.items() if isinstance(vecs, dict) else []):
        meta = getattr(v, "metadata", None) or (v.get("metadata") if isinstance(v, dict) else {})
        cid = (meta or {}).get("candidate_id") or vid.split("_chunk")[0]
        mapping.setdefault(cid, []).append(vid)
# print summary sorted by count desc
items = sorted(mapping.items(), key=lambda x: len(x[1]), reverse=True)
for cid, vids in items:
    print(f"{cid:50}  count={len(vids)}  sample_ids={vids[:3]}")
print("\nDone.")
