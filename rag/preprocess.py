"""Nettoyage des événements bruts Open Agenda.

Usage : uv run python -m rag.preprocess
"""

# --- IMPORT MODULES ----------------------------------

import html
import json
import re
from pathlib import Path

import pandas as pd

# --- CONSTANTES ----------------------------------

# Fichiers
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_PATH = DATA_DIR / "raw" / "events_toulouse.json"
PROCESSED_PATH = DATA_DIR / "processed" / "events_toulouse.json"

# Champs conservés (nom brut -> nom propre)
FIELDS = {
    "uid": "uid",
    "title_fr": "title",
    "description_fr": "description",
    "longdescription_fr": "long_description",
    "conditions_fr": "conditions",
    "keywords_fr": "keywords",
    "daterange_fr": "date_range",
    "firstdate_begin": "first_date",
    "lastdate_end": "last_date",
    "location_name": "location_name",
    "location_address": "location_address",
    "location_district": "location_district",
    "attendancemode": "attendance_mode",
    "status": "status",
    "originagenda_title": "agenda",
    "canonicalurl": "url",
}

# Champs texte
TEXT_FIELDS = ["title", "description", "long_description", "conditions", "location_district"]

# Date de début maximale (filtrage événements tests 2028)
MAX_START_DATE = "2027-12-31"

# --- FONCTIONS ----------------------------------

def clean_html(text: str) -> str:
    """Retire les balises HTML en gardant les retours à la ligne."""
    text = re.sub(r"<br\s*/?>|</p>|</li>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return re.sub(r"\n\s*\n+", "\n", text).strip()

def label_fr(value) -> str:
    """Extrait le libellé fr d'un champ JSON (status, attendancemode)."""
    if isinstance(value, str):
        value = json.loads(value)
    label = value["label"]
    return label["fr"] if isinstance(label, dict) else label

def preprocess(events: list[dict]) -> pd.DataFrame:
    """Sélectionne les champs utiles, supprime les événements inexploitables et nettoie les textes."""
    df = pd.DataFrame(events)[list(FIELDS)].rename(columns=FIELDS)
    df[TEXT_FIELDS] = df[TEXT_FIELDS].fillna("")
    print(f"Événements bruts : {len(df)}")

    # Suppression des événements inexploitables
    steps = {
        "sans titre": df["title"].str.strip() == "",
        "date lointaine": df["first_date"] > MAX_START_DATE,
        "annulés": df["status"].map(label_fr) == "Annulé",
        "doublon": df.duplicated(subset=["title", "first_date", "location_name"]),
    }
    for name, to_drop in steps.items():
        print(f"  - {to_drop.sum()} {name}")
    df = df[~pd.concat(steps, axis=1).any(axis=1)].copy()

    # Nettoyage des textes et champs structurés
    df["long_description"] = df["long_description"].map(clean_html)
    df["keywords"] = df["keywords"].map(lambda kw: list(dict.fromkeys(kw or [])))
    df["status"] = df["status"].map(label_fr)
    df["attendance_mode"] = df["attendance_mode"].map(label_fr)

    print(f"Événements conservés : {len(df)}")
    return df

# --- MAIN ----------------------------------

if __name__ == "__main__":
    events = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    df = preprocess(events)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(PROCESSED_PATH, orient="records", force_ascii=False, indent=2)
    print(f"Sauvegardé dans {PROCESSED_PATH}")
