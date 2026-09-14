"""Construction des documents à vectoriser à partir des événements nettoyés.

Usage : uv run python -m rag.documents
"""

# --- IMPORT MODULES ----------------------------------

import json
import re
from pathlib import Path

from langchain_core.documents import Document

# --- CONSTANTES ----------------------------------

# Fichiers
PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "events_toulouse.json"

# Champs gardés en métadonnées (non vectorisés)
METADATA_FIELDS = ["uid", "title", "date_range", "first_date", "last_date", "location_name", "status", "agenda", "url"]

# --- FONCTIONS ----------------------------------

def build_text(event: dict) -> str:
    """Construit le texte qui représente un événement pour vectorisation."""
    location = ", ".join(filter(None, [event["location_name"], event["location_address"], event["location_district"]]))

    # Description longue
    description = event["long_description"]

    # Description courte souvent reprise telle quelle dans la longue, on vérifie
    if event["description"][:80] not in description:
        description = f"{event['description']}\n{description}".strip()

    # Année absente de date_range pour l'année en cours : on l'ajoute pour lever l'ambiguïté
    dates = event["date_range"]
    if not re.search(r"\b20\d{2}\b", dates):
        dates = f"{dates} ({event['first_date'][:4]})"

    # Creation du contenu
    lines = [
        f"Titre : {event['title']}",
        f"Dates : {dates}",
        f"Lieu : {location}",
        f"Format : {event['attendance_mode']}" if event["attendance_mode"] != "Sur place" else "",
        f"Mots-clés : {', '.join(event['keywords'])}" if event["keywords"] else "",
        f"Description : {description}",
        f"Conditions : {event['conditions']}" if event["conditions"] else "",
    ]
    return "\n".join(line for line in lines if line)

def load_documents(path: Path = PROCESSED_PATH) -> list[Document]:
    """Charge les événements nettoyés et les convertit en Documents LangChain (texte + métadonnées)."""
    events = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(page_content=build_text(event), metadata={field: event[field] for field in METADATA_FIELDS})
        for event in events
    ]

# --- MAIN ----------------------------------

if __name__ == "__main__":
    documents = load_documents()
    print(f"{len(documents)} documents construits\n")

    # Visualisation d'un document
    print(documents[0].page_content)
    print("\nMétadonnées :", documents[0].metadata)

    # Statistiques
    lengths = sorted(len(doc.page_content) for doc in documents)
    print(f"\nLongueur des textes : médiane {lengths[len(lengths) // 2]}, max {lengths[-1]} caractères")
