"""Lancement local de l'API.

Usage : uv run python run.py [port]
"""

# --- IMPORT MODULES ----------------------------------

import os
import subprocess
import sys

from dotenv import load_dotenv

from rag.chain import INDEX_NAME
from rag.index import INDEX_DIR

# --- CONSTANTES ----------------------------------

PORT = "8000"

# --- MAIN ----------------------------------

if __name__ == "__main__":
    # Chargement env
    load_dotenv()
    port = sys.argv[1] if len(sys.argv) > 1 else PORT

    # Verif index
    if not (INDEX_DIR / INDEX_NAME / "index.faiss").exists():
        raise SystemExit(f"Index {INDEX_NAME} absent. Le reconstruire : uv run python -m rag.index {INDEX_NAME}")

    # Verif API key
    if not os.getenv("MISTRAL_API_KEY"):
        raise SystemExit("MISTRAL_API_KEY absente de l'environnement (voir .env.example)")

    # Verif token auth (warning uniquement)
    if not os.getenv("AUTH_TOKEN"):
        print("AUTH_TOKEN absent : /ask et /rebuild répondront 503 (voir .env.example)\n")

    print(f"Documentation interactive sur http://127.0.0.1:{port}/docs\n")
    subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app", "--port", port], check=False)
