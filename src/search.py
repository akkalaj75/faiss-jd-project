import os
import json
import faiss
import numpy as np
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


def search(query, index, model, chunks, meta, top_k=5):
    # Don’t ask for more neighbors than exist
    top_k = min(top_k, index.ntotal)

    q = model.encode([query]).astype("float32")
    faiss.normalize_L2(q)

    scores, ids = index.search(q, top_k)

    out = []
    for score, idx in zip(scores[0], ids[0]):
        # FAISS uses -1 when not enough results
        if idx == -1:
            continue
        out.append((float(score), meta[idx]["file"], meta[idx]["chunk_id"], chunks[idx]))
    return out


def main():
    chunks, meta, flat, ivf = load_assets()
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loaded index with {flat.ntotal} vectors.")
    print("Type your question. Type 'exit' to quit.")

    while True:
        q = input("\nAsk a question (type 'exit' to quit): ").strip()
        if q.lower() == "exit":
            break

        print("\n--- FLAT (Exact) ---")
        flat_results = search(q, flat, model, chunks, meta, top_k=5)
        for score, file, cid, text in flat_results:
            print(f"\nScore: {score:.4f} | {file} | chunk {cid}")
            print(text)

        print("\n--- IVF (Approx) ---")
        ivf_results = search(q, ivf, model, chunks, meta, top_k=5)
        for score, file, cid, text in ivf_results:
            print(f"\nScore: {score:.4f} | {file} | chunk {cid}")
            print(text)


if __name__ == "__main__":
    main()
