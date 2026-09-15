"""Tests des métriques simples de l'évaluation (rag/evaluate.py), sans appel API."""

# --- IMPORT MODULES ----------------------------------

from rag.evaluate import hit, hit_rate_by_category

# --- TESTS ----------------------------------

def test_hit():
    assert hit(["1", "2"], ["3", "2", "4"]) is True  # au moins un événement attendu retrouvé
    assert hit(["1"], ["3", "4"]) is False
    assert hit([], ["3", "4"]) is None  # aucun événement attendu : question non notée

def test_hit_rate_by_category():
    results = [
        {"category": "factuelle", "hit": True},
        {"category": "factuelle", "hit": False},
        {"category": "temporelle", "hit": True},
        {"category": "hors_sujet", "hit": None},  # ignorée
    ]
    assert hit_rate_by_category(results) == {"factuelle": 0.5, "temporelle": 1.0, "global": 2 / 3}
