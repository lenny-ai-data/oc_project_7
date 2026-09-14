"""Tests du nettoyage des événements (rag/preprocess.py)."""

# --- IMPORT MODULES ----------------------------------

from rag.preprocess import FIELDS, preprocess

# --- FONCTIONS ----------------------------------

def make_event(**overrides) -> dict:
    """Événement brut minimal, au format de l'API Open Agenda."""
    event = {field: None for field in FIELDS}
    event.update(
        uid="ok",
        title_fr="Concert",
        longdescription_fr="<p>Un concert &amp; un trio</p>",
        keywords_fr=["jazz", "jazz"],
        firstdate_begin="2026-06-20T18:00:00+00:00",
        location_name="Le Taquin",
        attendancemode='{"id": 1, "label": "Sur place"}',  # libellé non traduit
        status='{"id": 1, "label": {"fr": "Programmé", "en": "Scheduled"}}',  # libellé multilingue
    )
    event.update(overrides)
    return event

# --- TESTS ----------------------------------

def test_preprocess_filtering():
    events = [
        make_event(),
        make_event(uid="sans-titre", title_fr=""),
        make_event(uid="test", title_fr="test", firstdate_begin="2028-09-14T15:00:00+00:00"),
        make_event(uid="annule", title_fr="Théâtre", status='{"id": 6, "label": {"fr": "Annulé"}}'),
        make_event(uid="doublon"),
    ]
    assert preprocess(events)["uid"].tolist() == ["ok"]

def test_preprocess_cleaning():
    event = preprocess([make_event()]).iloc[0]
    assert event["long_description"] == "Un concert & un trio"
    assert event["keywords"] == ["jazz"]
    assert event["status"] == "Programmé"
    assert event["attendance_mode"] == "Sur place"
    assert event["conditions"] == ""
