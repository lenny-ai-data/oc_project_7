"""Fenêtre de chat locale pour la démo, qui s'ouvre dans le navigateur

Usage : uv run python scripts/chat_ui.py
"""

# --- IMPORT MODULES ----------------------------------

import base64
import os
from pathlib import Path

import gradio as gr
import requests

# Même dossier : URL de l'API (Render par défaut) et timeout
from ask_api import API_URL, TIMEOUT

# --- CONSTANTES ----------------------------------

ASSETS = Path(__file__).parent.parent / "assets"

# Username
DEMO_USER = "demo-puls-events"

# Message d'accueil
GREETING = "👋 Bonjour, je suis votre assistant de recherche d'évènements culturels sur Toulouse, est-ce que vous recherchez un type d'évènement en particulier ou sur une période précise ?"

# --- FONCTIONS ----------------------------------

def format_answer(result: dict) -> str:
    """Met en forme la réponse et ses sources en Markdown, titres cliquables"""
    lines = [result["answer"], "", "**Sources :**"]
    for source in result["sources"]:
        # Lieu et URL sont optionnels dans le modèle Source de l'API
        details = ", ".join(filter(None, [source["date_range"], source.get("location_name")]))
        title = f"[{source['title']}]({source['url']})" if source.get("url") else source["title"]
        lines.append(f"- {title} ({details})")
    return "\n".join(lines)


def login_logo() -> str:
    """Logo de la page de connexion, seul emplacement libre de la page : auth_message

    Image intégrée en base64
    """
    image = base64.b64encode((ASSETS / "Logo.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{image}" alt="Puls-Events" style="width:400px; margin:auto">'


def reply(message: str, history: list) -> str:
    """Envoie la question à l'API, une erreur s'affiche en bandeau dans la fenêtre"""
    response = requests.post(f"{API_URL}/ask", json={"question": message}, headers={"X-Token": os.getenv("AUTH_TOKEN")}, timeout=TIMEOUT)

    # FastAPI renvoie le motif de l'erreur dans detail
    if not response.ok:
        raise gr.Error(f"Erreur {response.status_code} : {response.json().get('detail')}")

    return format_answer(response.json())

# --- EXECUTION ----------------------------------

if __name__ == "__main__":
    # Verif token
    if not os.getenv("AUTH_TOKEN"):
        raise SystemExit("AUTH_TOKEN absent : renseigner le jeton dans le fichier .env")

    # Avatar chatbot
    chatbot = gr.Chatbot(avatar_images=(None, ASSETS / "favicon.png"), scale=1, value=[{"role": "assistant", "content": GREETING}])

    # Interface de chat
    demo = gr.ChatInterface(reply, chatbot=chatbot, title="Puls-Events", description=f"Assistant de recommandation d'événements culturels. API : {API_URL+"/docs"}")

    # PORT fourni par Render, 7860 en local
    demo.launch(favicon_path=ASSETS / "favicon.png", server_port=int(os.getenv("PORT", "7860")),
                auth=(DEMO_USER, os.getenv("AUTH_TOKEN")), auth_message=login_logo())
