"""Vérifie que l'environnement s'importe bien et que la clé API Mistral fonctionne

Usage : uv run python scripts/check_env.py
"""

# --- IMPORT MODULES ----------------------------------

import os

import faiss
from dotenv import load_dotenv
from langchain_community.vectorstores import (
    FAISS,  # noqa: F401 - importé pour vérifier que le paquet se charge
)
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# --- VERIF IMPORTS ----------------------------------

print(f"faiss {faiss.__version__} OK")
print(f"langchain-mistralai OK ({MistralAIEmbeddings.__name__}, {ChatMistralAI.__name__})")

# --- VERIF CLE API ----------------------------------

# Chargement du .env
load_dotenv()

if not os.getenv("MISTRAL_API_KEY"):
    raise SystemExit("MISTRAL_API_KEY absente : renseigner la clé dans le fichier .env")

# Test sur un appel réel (vectorisation d'une phrase)
vector = MistralAIEmbeddings(model="mistral-embed").embed_query("Concert de jazz à Toulouse")
print(f"API Mistral OK (embedding de dimension {len(vector)})")