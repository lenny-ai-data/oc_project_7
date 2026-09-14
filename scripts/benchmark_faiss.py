"""Compare les algorithmes d'index Faiss sur les vecteurs de l'index construit.

Aucun appel API : les questions sont simulées à partir des vecteurs du corpus.
Usage : uv run python scripts/benchmark_faiss.py
"""

# --- IMPORT MODULES ----------------------------------

import tempfile
import time
from pathlib import Path

import faiss
import numpy as np

# --- CONSTANTES ----------------------------------

INDEX_PATH = Path(__file__).parent.parent / "data" / "index" / "chunk_1000" / "index.faiss"

# Nombre de questions simulées
N_QUERIES = 500

# Nombre de résultats demandés
K = 5            

# Index FAISS comparés {nom: description}
''' 
- Flat : index plat "parfait"
- IVF (Inverted File) : regroupe les vecteurs en nlist groupe (k-means) et recherche dans les nprobe plus proches
- HNSW : Graphe de voisins sur plusieurs niveaux
- PQ : Quantification (dégrade la précision pour réduire la mémoire), se combine avec les autres techniques
'''
INDEXES = {
    "Flat (exact)": "Flat",
    "HNSW32": "HNSW32",
    "IVF80 (nprobe=8)": "IVF80,Flat",
    "IVF80,PQ64": "IVF80,PQ64",
}

# --- FONCTIONS ----------------------------------

def benchmark(factory: str, vectors: np.ndarray, queries: np.ndarray, truth: np.ndarray) -> dict:
    """Construit un index, mesure la construction, la recherche, le rappel et la taille."""
    index = faiss.index_factory(vectors.shape[1], factory)

    start = time.perf_counter()
    # Sans effet pour Flat et HNSW, calcul des groupes et compression pour IVF
    index.train(vectors)
    index.add(vectors)
    build_ms = (time.perf_counter() - start) * 1000

    if "IVF" in factory:
        faiss.extract_index_ivf(index).nprobe = 8  # nombre de groupes visités par recherche

    start = time.perf_counter()
    _, found = index.search(queries, K)
    search_ms = (time.perf_counter() - start) / len(queries) * 1000

    # Rappel@K : part des K vrais plus proches voisins (recherche exacte) retrouvés
    recall = np.mean([len(set(f) & set(t)) / K for f, t in zip(found, truth)])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.faiss"
        faiss.write_index(index, str(path))
        size_mb = path.stat().st_size / 1e6

    return {"build_ms": build_ms, "search_ms": search_ms, "recall": recall, "size_mb": size_mb}

# --- MAIN ----------------------------------

if __name__ == "__main__":
    # Vecteurs de l'index construit par rag.index
    index = faiss.read_index(str(INDEX_PATH))
    vectors = index.reconstruct_n(0, index.ntotal)
    print(f"{len(vectors)} vecteurs de dimension {vectors.shape[1]}\n")

    # Questions simulées : vecteurs du corpus légèrement bruités
    rng = np.random.default_rng(0)
    queries = vectors[rng.choice(len(vectors), N_QUERIES, replace=False)]
    queries = queries + rng.normal(0, 0.01, queries.shape).astype("float32")

    # Référence : les vrais plus proches voisins (recherche exhaustive)
    exact = faiss.IndexFlatL2(vectors.shape[1])
    exact.add(vectors)
    _, truth = exact.search(queries, K)

    for name, factory in INDEXES.items():
        r = benchmark(factory, vectors, queries, truth)
        print(f"{name:<18} construction {r['build_ms']:7.0f} ms | recherche {r['search_ms']:.3f} ms/question"
              f" | rappel@{K} {r['recall']:.0%} | {r['size_mb']:5.1f} Mo")
