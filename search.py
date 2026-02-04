import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = "indexes"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_assets():
    with open(os.path.join(INDEX_DIR, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(os.path.join(INDEX_DIR, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    flat = faiss.read_index(os.path.join(INDEX_DIR, "flat.index"))
    ivf = faiss.read_index(os.path.join(INDEX_DIR, "ivf.index"))

    return chunks, meta, flat, ivf

def run_search(query, index, model, chunks, meta, top_k=5):
    q = model.encode([query]).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append({
            "score": float(score),
            "file": meta[idx]["file"],
            "chunk_id": meta[idx]["chunk_id"],
            "text": chunks[idx]
        })
    return results

def main():
    chunks, meta, flat, ivf = load_assets()
    model = SentenceTransformer(MODEL_NAME)

    while True:
        q = input("\nAsk a question (type 'exit' to quit): ").strip()
        if q.lower() == "exit":
            break

        print("\n--- FLAT (Exact) ---")
        for r in run_search(q, flat, model, chunks, meta):
            print(f"\nScore: {r['score']:.4f} | {r['file']} | chunk {r['chunk_id']}")
            print(r["text"])

        print("\n--- IVF (Approx) ---")
        for r in run_search(q, ivf, model, chunks, meta):
            print(f"\nScore: {r['score']:.4f} | {r['file']} | chunk {r['chunk_id']}")
            print(r["text"])

if __name__ == "__main__":
    main()
