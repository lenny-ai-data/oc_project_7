"""Évaluation du système RAG sur le jeu de test annoté.

Usage :
    uv run python -m rag.evaluate run [nom_index]     pose les questions au RAG, hit@5, sauvegarde des résultats
    uv run python -m rag.evaluate ragas [nom_index]   scores Ragas à partir des résultats sauvegardés
"""

# --- IMPORT MODULES ----------------------------------

import importlib
import json
import math
import sys
import types
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

from langchain_mistralai import ChatMistralAI

from rag.chain import INDEX_NAME, RAG
from rag.index import INDEX_DIR, get_embeddings, load_index

# --- CONSTANTES ----------------------------------

# Fichiers
EVAL_DIR = Path(__file__).parent.parent / "eval"
TEST_SET_PATH = EVAL_DIR / "test_set.json"
RESULTS_DIR = EVAL_DIR / "results"

# LLM juge pour Ragas (188 req/min en free plan)
JUDGE_MODEL = "ministral-8b-latest"

# Métriques Ragas, dans l'ordre des colonnes produites
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]

# --- FONCTIONS ----------------------------------

def hit(expected_uids: list[str], source_uids: list[str]) -> bool | None:
    """Vrai si au moins un événement attendu fait partie des sources (None si aucun événement attendu)."""
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

def patch_ragas_import() -> None:
    """Contournement d'un bug de Ragas 0.4.3 (https://github.com/vibrantlabsai/ragas/issues/2745).

    Ragas importe ChatVertexAI depuis un module retiré de langchain-community 0.4"""

    # Module cible
    VERTEXAI_MODULE = "langchain_community.chat_models.vertexai"

    try:
        importlib.import_module(VERTEXAI_MODULE)
    except ModuleNotFoundError:
        stub = types.ModuleType(VERTEXAI_MODULE)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[VERTEXAI_MODULE] = stub

def ragas_scores(results: list[dict]) -> list[dict]:
    """Ajoute les scores Ragas aux questions qui attendent des événements (les autres n'ont pas de bons contextes)."""
    # --- Import Ragas ----------------------------------
    patch_ragas_import()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    # ---------------------------------------------------

    # Selection des questions avec une réponse attendue
    rated = [result for result in results if result["expected_uids"]]

    # Préparation au format Ragas
    dataset = EvaluationDataset(samples=[
        SingleTurnSample(user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"], reference=r["reference"])
        for r in rated
    ])

    # Chargement du modele d'embeddings
    embeddings = get_embeddings()

    # Evaluation (9 appels LLM par question)
    scores = evaluate(
        dataset,
        # strictness=1 (Ragas échoue en fusionnant les réponses de ChatMistralAI)
        metrics=[Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=LangchainLLMWrapper(ChatMistralAI(model=JUDGE_MODEL, temperature=0)),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=RunConfig(max_workers=4, timeout=180),
    )

    # Scores ajoutés à chaque résultat (NaN remplacé par None pour un JSON valide)
    for result, row in zip(rated, scores.to_pandas()[RAGAS_METRICS].to_dict("records")):
        result["ragas"] = {metric: (None if math.isnan(value) else round(value, 3)) for metric, value in row.items()}
    return results

def ragas_means(results: list[dict]) -> dict[str, float]:
    """Moyenne de chaque métrique Ragas sur les questions évaluées."""
    means = {}
    for metric in RAGAS_METRICS:
        values = [r["ragas"][metric] for r in results if r.get("ragas") and r["ragas"][metric] is not None]
        means[metric] = sum(values) / len(values) if values else None
    return means

# --- MAIN ----------------------------------

if __name__ == "__main__":
    # Commande et nom de l'index
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    index_name = sys.argv[2] if len(sys.argv) > 2 else INDEX_NAME
    results_path = RESULTS_DIR / f"{index_name}.json"

    if command == "run":
        # Questionnaire de test
        test_set = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))

        # Soumission du questionnaire et évaluation des hits
        results = run_test_set(RAG(index=load_index(INDEX_DIR / index_name)), test_set)

        # Résultats
        print(f"\nHit@5 ({index_name}) :")
        for category, rate in hit_rate_by_category(results).items():
            print(f"  {category:<20} {rate:.0%}")
        print("\nÉchecs :", [r["id"] for r in results if r["hit"] is False])

    elif command == "ragas":
        # Scores Ragas à partir des réponses déjà sauvegardées (sans rappeler le RAG)
        results = ragas_scores(json.loads(results_path.read_text(encoding="utf-8")))

        # Résultats
        print(f"\nRagas ({index_name}), moyennes :")
        for metric, mean in ragas_means(results).items():
            print(f"  {metric:<40} {mean:.2f}" if mean is not None else f"  {metric:<40} -")

    else:
        raise SystemExit("Commande inconnue : utiliser « run » ou « ragas »")

    # Sauvegarde des réponses, sources, contextes et scores
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats sauvegardés dans {results_path}")
