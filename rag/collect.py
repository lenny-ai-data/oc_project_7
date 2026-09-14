"""Collecte des événements Open Agenda de Toulouse (dataset OpenDataSoft).

Usage : uv run python -m rag.collect
"""

# --- IMPORT MODULES ----------------------------------

import json
from pathlib import Path

import requests

# --- CONSTANTES ----------------------------------

# API
EXPORT_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/exports/json"

# Filtres
CITY = "Toulouse"
START_DATE = "2025-09-01"

# Agendas exclus (sans rapport avec la culture)
EXCLUDED_AGENDAS = {
    "38495884": "Mes événements France Travail",
    "38376650": "Challenges Geovelo",
    "9464342": "Semaine de l'industrie 2025",
    "17542672": "TM - Renov'energie",
}

# Répertoire de stockage des données
RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "events_toulouse.json"

# --- FONCTIONS ----------------------------------

def build_where() -> str:
    """Construit le filtre ODSQL : ville, période et agendas exclus."""
    conditions = [f"location_city = '{CITY}'", f"lastdate_end >= date'{START_DATE}'"]
    conditions += [f"originagenda_uid != '{uid}'" for uid in EXCLUDED_AGENDAS]
    return " AND ".join(conditions)

def fetch_events() -> list[dict]:
    """Télécharge tous les événements filtrés"""
    response = requests.get(EXPORT_URL, params={"where": build_where()}, timeout=120)
    response.raise_for_status()
    return response.json()

def save_events(events: list[dict], path: Path = RAW_PATH) -> None:
    """Sauvegarde les événements bruts en JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

# --- MAIN ----------------------------------

if __name__ == "__main__":
    events = fetch_events()
    save_events(events)
    print(f"{len(events)} événements sauvegardés dans {RAW_PATH}")
