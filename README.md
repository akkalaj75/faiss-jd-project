# FAISS Semantic Search Project (Flat + IVF)

This repository demonstrates an end-to-end **semantic search system** built using **FAISS**, aligned with real-world **AI / Machine Learning Developer** job requirements.

The project showcases document ingestion, vector embedding, exact vs approximate similarity search, and performance benchmarking.

---

## 🚀 Key Features

- Safe document ingestion and chunking
- SentenceTransformers-based embeddings
- FAISS indexing:
  - **IndexFlatIP** (exact similarity search)
  - **IndexIVFFlat** (approximate similarity search)
- Interactive semantic search (CLI)
- Latency benchmarking with `nprobe` tuning
- Production-style project structure (`src/` based)

---

## 🧠 Tech Stack

- Python
- FAISS (CPU)
- SentenceTransformers
- NumPy

---

## 📂 Project Structure

```text
faiss-jd-project/
├── docs/              # Input text documents
├── indexes/           # Generated FAISS indexes (ignored in git)
├── src/
│   ├── ingest.py      # Document ingestion & chunking
│   ├── build_index.py # Embedding generation & FAISS index creation
│   ├── search.py      # Semantic search CLI
│   └── evaluate.py    # Performance benchmarking
├── .gitignore
└── README.md
▶️ How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Build FAISS indexes
python src/build_index.py

3️⃣ Run semantic search
python src/search.py

4️⃣ Benchmark performance
python src/evaluate.py

📊 What This Project Demonstrates

Vector similarity search using embeddings

FAISS index selection and tuning (nlist, nprobe)

Exact vs approximate retrieval trade-offs

Latency benchmarking for scalable AI systems

Clean, reproducible ML project design

🎯 Use Cases

Knowledge base search

Internal documentation retrieval

AI-powered question answering systems

Recommendation and similarity engines

👤 Author

Jyothin
