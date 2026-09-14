"""Tests de la collecte Open Agenda (rag/collect.py)."""

# --- IMPORT MODULES ----------------------------------

import json

import pytest

from rag import collect

# --- TESTS ----------------------------------

def test_build_where_filters():
    where = collect.build_where()
    assert f"location_city = '{collect.CITY}'" in where
    assert f"lastdate_end >= date'{collect.START_DATE}'" in where
    for uid in collect.EXCLUDED_AGENDAS:
        assert f"originagenda_uid != '{uid}'" in where

@pytest.mark.skipif(not collect.RAW_PATH.exists(), reason="Données brutes absentes (lancer rag.collect)")
def test_current_raw_data():
    events = json.loads(collect.RAW_PATH.read_text(encoding="utf-8"))
    assert events
    assert all(ev["location_city"] == collect.CITY for ev in events)
    assert all(ev["lastdate_end"] >= collect.START_DATE for ev in events)
    assert not any(ev["originagenda_uid"] in collect.EXCLUDED_AGENDAS for ev in events)
