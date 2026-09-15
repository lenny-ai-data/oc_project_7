"""Vectorisation des documents et index Faiss (création, sauvegarde, chargement).

Usage :
    uv run python -m rag.index              toutes les configurations
    uv run python -m rag.index chunk_1000   une seule configuration
"""

# --- IMPORT MODULES ----------------------------------

import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

from rag.documents import load_documents, split_documents

# --- CONSTANTES ----------------------------------

# Dossier de sauvegarde
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"

# Modèle d'embedding Mistral (dim. 1024)
EMBEDDING_MODEL = "mistral-embed"

# Configurations comparées : nom du dossier -> (chunk_size, chunk_overlap)
CONFIGS = {
    "no_chunk": (None, 0),
    "chunk_1000": (1000, 150),
}

# --- FONCTIONS ----------------------------------

def get_embeddings() -> MistralAIEmbeddings:
    """Crée le client d'embedding Mistral.
    
    Gère directement le traitement par batch de 16k tokens.
    """
    load_dotenv()
    return MistralAIEmbeddings(model=EMBEDDING_MODEL)

def build_index(documents: list[Document]) -> FAISS:
    """Vectorise les documents et les range dans un index Faiss."""
    return FAISS.from_documents(documents, get_embeddings())

def save_index(index: FAISS, path: Path = INDEX_DIR) -> None:
    """Sauvegarde l'index sur disque (vecteurs + textes et métadonnées).

    Fichier .faiss contenant les vecteurs
    Fichier .pkl textes et métadonnées    
    """
    index.save_local(str(path))

def load_index(path: Path = INDEX_DIR) -> FAISS:
    """Recharge un index sauvegardé associé au modèle d'embeddings."""
    return FAISS.load_local(str(path), get_embeddings(), allow_dangerous_deserialization=True)

# --- MAIN ----------------------------------

if __name__ == "__main__":
    # Configurations à construire : celle passée en argument, sinon toutes
    names = sys.argv[1:] or list(CONFIGS)
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise SystemExit(f"Configuration inconnue : {unknown}. Choix possibles : {list(CONFIGS)}")

    # Chargement des documents
    documents = load_documents()

    # Un index par configuration de découpage
    for name in names:
        chunk_size, chunk_overlap = CONFIGS[name]
        # Chunking
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        print(f"\n[{name}] {len(chunks)} textes à vectoriser...")

        # Vectorisation
        start = time.perf_counter()
        index = build_index(chunks)
        save_index(index, INDEX_DIR / name)
        print(f"[{name}] {index.index.ntotal} vecteurs de dimension {index.index.d} en {time.perf_counter() - start:.0f} s")

    # Vérification : rechargement et recherche sur le dernier index
    index = load_index(INDEX_DIR / name)
    for doc, score in index.similarity_search_with_score("visite guidée au musée", k=3):
        print(f"[distance {score:.3f}] {doc.metadata['title']} - {doc.metadata['date_range']}")
