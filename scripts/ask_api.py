"""Pose une question à l'API déployée et affiche la réponse puis les sources

Usage : uv run python scripts/ask_api.py "Je cherche une pièce de théâtre, tu as des idées ?"
        (API_URL=http://localhost:8000 pour viser l'API locale)
"""

# --- IMPORT MODULES ----------------------------------

import os
import sys

import requests
from dotenv import load_dotenv

# --- CONSTANTES ----------------------------------

# AUTH_TOKEN, et éventuellement API_URL
load_dotenv()

API_URL = os.getenv("API_URL", "https://puls-events-api.onrender.com")

# L'instance gratuite Render met ~1 min à se réveiller
TIMEOUT = 120

# --- FONCTIONS ----------------------------------

def print_answer(question: str, result: dict) -> None:
    """Affiche la réponse, puis les sources avec leur lien"""
    print(f"Question : {question}\n")
    print(result["answer"])
    print("\nSources :")
    for source in result["sources"]:
        # Lieu et URL sont optionnels dans le modèle Source de l'API
        details = ", ".join(filter(None, [source["date_range"], source.get("location_name")]))
        print(f"- {source['title']} ({details})")
        if source.get("url"):
            print(f"  {source['url']}")

# --- EXECUTION ----------------------------------

if __name__ == "__main__":
    # Jointure des arguments : les guillemets autour de la question sont facultatifs
    question = " ".join(sys.argv[1:])
    if not question:
        raise SystemExit(__doc__)

    token = os.getenv("AUTH_TOKEN")
    if not token:
        raise SystemExit("AUTH_TOKEN absent : renseigner le jeton dans le fichier .env")

    print(f"Question envoyée à {API_URL}...\n")
    response = requests.post(f"{API_URL}/ask", json={"question": question},
                             headers={"X-Token": token}, timeout=TIMEOUT)

    # FastAPI renvoie le motif de l'erreur dans « detail » (401, 422, 502, 503)
    if not response.ok:
        raise SystemExit(f"Erreur {response.status_code} : {response.json().get('detail')}")

    print_answer(question, response.json())
