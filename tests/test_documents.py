"""Tests de la construction des documents (rag/documents.py)."""

# --- IMPORT MODULES ----------------------------------

import json

from rag.documents import METADATA_FIELDS, build_text, load_documents

# --- CONSTANTES ----------------------------------

# Événement nettoyé minimal
EVENT = {
    "uid": "1",
    "title": "Concert de jazz",
    "description": "Un concert.",
    "long_description": "Un concert.\nAvec un trio.",
    "conditions": "",
    "keywords": [],
    "date_range": "Samedi 20 juin, 20h00",
    "first_date": "2026-06-20T18:00:00+00:00",
    "last_date": "2026-06-20T20:00:00+00:00",
    "location_name": "Le Taquin",
    "location_address": "23 rue Ingres",
    "location_district": "",
    "attendance_mode": "Sur place",
    "status": "Programmé",
    "agenda": "Agenda",
    "url": "https://openagenda.com/events/1",
}

# --- TESTS ----------------------------------

def test_build_text():
    text = build_text(EVENT)
    assert "Titre : Concert de jazz" in text
    assert "Lieu : Le Taquin, 23 rue Ingres" in text
    assert "Dates : Samedi 20 juin, 20h00 (2026)" in text
    assert text.count("Un concert.") == 1  # description courte non répétée
    assert "Conditions" not in text

def test_load_documents(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps([EVENT]), encoding="utf-8")

    [doc] = load_documents(path)

    assert doc.page_content == build_text(EVENT)
    assert set(doc.metadata) == set(METADATA_FIELDS)
