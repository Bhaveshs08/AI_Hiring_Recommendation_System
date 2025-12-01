# inspect_vectors.py — run with: python inspect_vectors.py
import os, json
from pinecone import Pinecone

api = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX")
if not api or not index_name:
    raise SystemExit("Set PINECONE_API_KEY and PINECONE_INDEX in environment.")

pc = Pinecone(api_key=api)
idx = pc.Index(index_name)

# Adjust these if your IDs differ
resume_ns = "Resumes"
jd_ns = "Job_Descriptions"

# List up to 10 ids in each namespace
print("Listing sample vectors via index.list()...")
res_list = list(idx.list(namespace=resume_ns, limit=10))
jd_list = list(idx.list(namespace=jd_ns, limit=10))

print("\nResumes list (repr):")
for item in res_list:
    print(repr(item)[:1000])
print("\nJDs list (repr):")
for item in jd_list:
    print(repr(item)[:1000])

# If ids were printed above, extract them; otherwise you can hardcode
res_ids = []
jd_ids = []
for item in res_list:
    if isinstance(item, dict):
        if item.get("id"):
            res_ids.append(item.get("id"))
    else:
        # try attrs
        if hasattr(item, "id"):
            res_ids.append(getattr(item, "id"))
for item in jd_list:
    if isinstance(item, dict):
        if item.get("id"):
            jd_ids.append(item.get("id"))
    else:
        if hasattr(item, "id"):
            jd_ids.append(getattr(item, "id"))

print("\nSample resume ids:", res_ids)
print("Sample jd ids:", jd_ids)

# Fetch the first resume and JD (if present)
def pretty_fetch(ids, namespace, label):
    if not ids:
        print(f"No ids to fetch for {label}")
        return
    print(f"\nFetching {label} ids: {ids}")
    fetched = idx.fetch(ids=ids, namespace=namespace)
    print("Fetch repr (first 4000 chars):")
    print(repr(fetched)[:4000])
    # Try to to_dict() if available
    try:
        d = fetched.to_dict()
        print("Fetched .to_dict() keys:", list(d.keys()))
        if "vectors" in d:
            for k,v in d["vectors"].items():
                print(f"ID {k} -> keys: {list(v.keys())}")
    except Exception as e:
        print("to_dict() not available or failed:", e)

pretty_fetch(res_ids[:5], resume_ns, "Resumes")
pretty_fetch(jd_ids[:5], jd_ns, "Job_Descriptions")
