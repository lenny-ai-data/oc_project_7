"""Construction des documents à vectoriser à partir des événements nettoyés.

Usage : uv run python -m rag.documents
"""

# --- IMPORT MODULES ----------------------------------

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def split_documents(documents: list[Document], chunk_size: int | None, chunk_overlap: int = 0) -> list[Document]:
    """Découpe les documents trop longs en chunks, en répétant l'en-tête (titre, dates, lieu) dans chacun.

    chunk_size=None : pas de découpage, un événement = un vecteur.
    """
    if chunk_size is None:
        return documents

    chunks = []
    for doc in documents:
        if len(doc.page_content) <= chunk_size:
            chunks.append(doc)
            continue

        # En-tête jusqu'à la fin de la ligne "Lieu"
        head, _, rest = doc.page_content.partition("\nLieu : ")
        location, _, body = rest.partition("\n")
        header = f"{head}\nLieu : {location}"

        # Conditions (tarif, réservation), en fin de texte : déplacées dans l'en-tête pour figurer dans chaque chunk
        if "\nConditions : " in body:
            body, _, conditions = body.rpartition("\nConditions : ")
            header += f"\nConditions : {conditions}"

        # Le corps est découpé avec insert de l'en-tête dans chaque chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size - len(header) - 1, chunk_overlap=chunk_overlap)
        for part in splitter.split_text(body):
            chunks.append(Document(page_content=f"{header}\n{part}", metadata=doc.metadata))
    return chunks

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

    # Découpage : exemple sur le document le plus long
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=150)
    print(f"\nDécoupage à 1000 caractères : {len(chunks)} chunks")

    longest = max(documents, key=lambda doc: len(doc.page_content))
    longest_chunks = split_documents([longest], chunk_size=1000, chunk_overlap=150)
    print(f"Document le plus long -> {len(longest_chunks)} chunks, dont les 2 premiers :\n")
    
    for chunk in longest_chunks[:2]:
        print(chunk.page_content, "\n---")
