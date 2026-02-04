import os
import time
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = "indexes"
MODEL_NAME = "all-MiniLM-L6-v2"


def benchmark(index, queries, model, k=5):
    # cap k so we don't ask more than exists
    k = min(k, index.ntotal)

    t0 = time.time()
    for q in queries:
        q_emb = model.encode([q]).astype("float32")
        faiss.normalize_L2(q_emb)
        index.search(q_emb, k)
    t1 = time.time()

    return (t1 - t0) / len(queries) * 1000  # ms/query


def main():
    model = SentenceTransformer(MODEL_NAME)
    flat = faiss.read_index(os.path.join(INDEX_DIR, "flat.index"))
    ivf = faiss.read_index(os.path.join(INDEX_DIR, "ivf.index"))

    queries = [
        "VPN troubleshooting steps",
        "password policy requirements",
        "cloud deployment checklist",
        "MFA for remote access",
        "Docker CI CD guidelines"
    ]

    flat_ms = benchmark(flat, queries, model, k=5)
    print(f"\n📊 Benchmark (ms/query)")
    print(f"Flat (Exact): {flat_ms:.2f} ms/query")

    print("\nIVF (Approx) latency by nprobe (speed vs accuracy tradeoff):")
    for nprobe in [1, 3, 5, 10, 20]:
        ivf.nprobe = min(nprobe, getattr(ivf, "nlist", nprobe))
        ivf_ms = benchmark(ivf, queries, model, k=5)
        print(f"  IVF nprobe={ivf.nprobe:<2}: {ivf_ms:.2f} ms/query")

    print(f"\nIndex size: {flat.ntotal} vectors")
    print("Tip: Add more docs to /docs and rebuild index to see IVF benefits clearly.")


if __name__ == "__main__":
    main()
