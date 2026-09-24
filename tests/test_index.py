"""Tests des index vectoriels construits (rag/index.py)."""

# --- IMPORT MODULES ----------------------------------

import os
from datetime import datetime

import pytest
from dotenv import load_dotenv

from rag.documents import METADATA_FIELDS, load_documents, split_documents
from rag.index import CONFIGS, INDEX_DIR, TIMEZONE, built_at, load_index

# --- CONSTANTES ----------------------------------

# Test possible seulement si les index existent et que la clé Mistral est disponible (vectorisation de la question)
load_dotenv()
INDEX_READY = all((INDEX_DIR / name).exists() for name in CONFIGS) and bool(os.getenv("MISTRAL_API_KEY"))

# --- TESTS ----------------------------------

@pytest.mark.skipif(not INDEX_READY, reason="Index ou clé Mistral absents (lancer rag.index)")
def test_current_indexes():
    documents = load_documents()

    for name, (chunk_size, chunk_overlap) in CONFIGS.items():
        index = load_index(INDEX_DIR / name)

        # Tous les chunks sont indexés, avec la dimension de mistral-embed
        assert index.index.ntotal == len(split_documents(documents, chunk_size, chunk_overlap))
        assert index.index.d == 1024

        # Cas connu : le titre exact d'un événement le renvoie en premier, avec ses métadonnées
        [doc] = index.similarity_search("Visite libre de la basilique Saint-Sernin", k=1)
        assert doc.metadata["title"] == "Visite libre de la basilique Saint-Sernin"
        assert set(doc.metadata) == set(METADATA_FIELDS)

def test_built_at(tmp_path):
    # Repli sur la date du fichier de vecteurs quand la date n'a pas été écrite
    (tmp_path / "index.faiss").write_bytes(b"")
    assert built_at(tmp_path) == datetime.now(TIMEZONE).date().isoformat()

    # Date écrite à la sauvegarde : elle survit à un git clone, la date de fichier non
    (tmp_path / "build_info.json").write_text('{"built_at": "2026-09-15"}', encoding="utf-8")
    assert built_at(tmp_path) == "2026-09-15"
