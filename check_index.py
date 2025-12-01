#!/usr/bin/env python3
import os, sys, json
from pinecone import Pinecone

api = os.environ.get("PINECONE_API_KEY")
idx = os.environ.get("PINECONE_INDEX")
if not api or not idx:
    print("Missing PINECONE_API_KEY or PINECONE_INDEX environment variables.")
    sys.exit(2)

pc = Pinecone(api_key=api)
pc_index = pc.Index(idx)

try:
    stats = pc_index.describe_index_stats()
    print("Index name:", idx)
    print("Dimension:", stats.get("dimension"))
    print("Total vectors:", stats.get("total_vector_count"))
    # show namespaces counts if provided
    namespaces = stats.get("namespaces") or {}
    if namespaces:
        print("Namespaces and counts:")
        for ns, meta in namespaces.items():
            print(f"  {ns} : {meta.get('vector_count')}")
    else:
        print("No namespace breakdown available in describe_index_stats() output.")
    # quick list of namespace presence
    print("\nQuick existence check for expected namespaces:")
    for ns in ('Resumes','Job_Descriptions'):
        try:
            res = pc_index.describe_index_stats(namespace=ns)
            print(f"  {ns}: ok (may return namespace stats depending on account)")
        except Exception:
            print(f"  {ns}: could not query namespace stats (SDK behaviour)")
except Exception as e:
    print("Error querying index:", e)
    sys.exit(1)

