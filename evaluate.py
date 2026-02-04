import os
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = "indexes"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_indexes():
    flat = faiss.read_index(os.path.join(INDEX_DIR, "flat.index"))
    ivf = faiss.read_index(os.path.join(INDEX_DIR, "ivf.index"))
    return flat, ivf

def benchmark(index, queries, model, k=5):
    start = time.time()
    for q in queries:
        q_emb = model.encode([q]).astype("float32")
        faiss.normalize_L2(q_emb)
        index.search(q_emb, k)
    end = time.time()
    return (end - start) / len(queries) * 1000  # ms/query

def main():
    model = SentenceTransformer(MODEL_NAME)
    flat, ivf = load_indexes()

    queries = [
        "VPN troubleshooting steps",
        "password policy requirements",
        "cloud deployment best practices",
        "MFA for remote access",
        "Docker CI CD guidelines"
    ]

    flat_ms = benchmark(flat, queries, model)
    print(f"\nFlat (Exact) latency: {flat_ms:.2f} ms/query")

    print("\nIVF latency by nprobe (speed vs accuracy tradeoff):")
    for nprobe in [1, 5, 10, 20]:
        ivf.nprobe = nprobe
        ivf_ms = benchmark(ivf, queries, model)
        print(f"  IVF nprobe={nprobe:<2}: {ivf_ms:.2f} ms/query")

if __name__ == "__main__":
    main()
