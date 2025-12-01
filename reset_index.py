import os
from pinecone import Pinecone

# ====== CONFIG ======
INDEX_NAME = "polaris"        # target index
DIMENSION = 3072              # must match embeddings (text-embedding-3-large)
METRIC = "cosine"
NAMESPACE = "resumes"
# ====================

# Init Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def reset_index():
    try:
        # Try deleting old index if exists
        try:
            pc.delete_index(INDEX_NAME)
            print(f"🗑️ Deleted old index: {INDEX_NAME}")
        except Exception as e:
            print(f"⚠️ No existing index deleted ({e})")

        # Create fresh index
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )
        print(f"✅ Created fresh index: {INDEX_NAME} ({DIMENSION}-dim, {METRIC})")

    except Exception as e:
        print(f"❌ Failed to reset index: {e}")


if __name__ == "__main__":
    reset_index()
