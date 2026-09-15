"""Évaluation du système RAG sur le jeu de test annoté.

Usage : uv run python -m rag.evaluate [nom_index]   (no_chunk par défaut)
"""

# --- IMPORT MODULES ----------------------------------

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from rag.chain import INDEX_NAME, RAG
from rag.index import INDEX_DIR, load_index

# --- CONSTANTES ----------------------------------

# Fichiers
EVAL_DIR = Path(__file__).parent.parent / "eval"
TEST_SET_PATH = EVAL_DIR / "test_set.json"
RESULTS_DIR = EVAL_DIR / "results"

# --- FONCTIONS ----------------------------------

def hit(expected_uids: list[str], source_uids: list[str]) -> bool | None:
    """Vrai si au moins un événement attendu fait partie des sources (None si aucun événement n'est attendu)."""
    if not expected_uids:
        return None
    return bool(set(expected_uids) & set(source_uids))

def run_test_set(rag: RAG, test_set: dict) -> list[dict]:
    """Pose chaque question au RAG à la date de référence et conserve la réponse, les sources et les contextes."""
    today = date.fromisoformat(test_set["reference_date"])
    results = []
    for i, item in enumerate(test_set["questions"], start=1):
        print(f"[{i}/{len(test_set['questions'])}] {item['id']}")
        output = rag.ask(item["question"], today=today)
        source_uids = [source["uid"] for source in output["sources"]]
        results.append({
            **item,
            "answer": output["answer"],
            "source_uids": source_uids,
            "contexts": output["contexts"],
            "hit": hit(item["expected_uids"], source_uids),
        })
    return results

def hit_rate_by_category(results: list[dict]) -> dict[str, float]:
    """Taux de hit par catégorie et global, sur les questions qui attendent des événements."""
    hits = defaultdict(list)
    for result in results:
        if result["hit"] is not None:
            hits[result["category"]].append(result["hit"])
            hits["global"].append(result["hit"])
    return {category: sum(values) / len(values) for category, values in hits.items()}

# --- MAIN ----------------------------------

if __name__ == "__main__":
    # Nom de l'index
    index_name = sys.argv[1] if len(sys.argv) > 1 else INDEX_NAME

    # Questionnaire de test
    test_set = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))

    # Soumission du questionnaire et évaluation des hits
    results = run_test_set(RAG(index=load_index(INDEX_DIR / index_name)), test_set)

    # Sauvegarde des réponses, sources et contexte
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{index_name}.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats sauvegardés dans {results_path}")

    # Résultats
    print(f"\nHit@5 ({index_name}) :")
    for category, rate in hit_rate_by_category(results).items():
        print(f"  {category:<20} {rate:.0%}")
    print("\nÉchecs :", [r["id"] for r in results if r["hit"] is False])
