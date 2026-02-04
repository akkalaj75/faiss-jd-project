import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ingest import load_docs

INDEX_DIR = "indexes"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Loading documents...")
    chunks, meta = load_docs("docs")
    print(f"Loaded {len(chunks)} chunks")

    if len(chunks) == 0:
        raise ValueError("No .txt files found in /docs. Add at least one .txt file and rerun.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize to use cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]

    # ---------- 1) Flat index (exact search) ----------
    flat_index = faiss.IndexFlatIP(d)
    flat_index.add(embeddings)

    # ---------- 2) IVF index (approx search) ----------
    # nlist must be <= number of vectors (training points)
    n_vectors = embeddings.shape[0]
    # nlist = max(1, min(50, n_vectors // 2))
    # Better heuristic for IVF: nlist around sqrt(N)
    nlist = int(np.sqrt(n_vectors))
    nlist = max(4, min(64, nlist))  # clamp to sane bounds


    quantizer = faiss.IndexFlatIP(d)
    ivf_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train only if meaningful (FAISS can still train with small sizes but warns)
    ivf_index.train(embeddings)
    ivf_index.add(embeddings)

    # nprobe controls speed vs accuracy
    # ivf_index.nprobe = min(10, nlist)
# nprobe ~ 10-30% of nlist is a common starting point
    ivf_index.nprobe = max(1, min(nlist, nlist // 4))


    # ---------- Save ----------
    faiss.write_index(flat_index, os.path.join(INDEX_DIR, "flat.index"))
    faiss.write_index(ivf_index, os.path.join(INDEX_DIR, "ivf.index"))

    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    with open(os.path.join(INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Indexes saved successfully")
    print(f"Flat vectors: {flat_index.ntotal} | IVF vectors: {ivf_index.ntotal} | IVF nlist: {nlist} | IVF nprobe: {ivf_index.nprobe}")


if __name__ == "__main__":
    main()
