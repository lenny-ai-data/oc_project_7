# Itérations d'évaluation

Jeu de test : `eval/test_set.json` (20 questions, date de référence 15/09/2026).
Hit@5 sur les 15 questions qui attendent des événements ; scores Ragas (juge `ministral-8b`) sur ces mêmes 15 questions.
Les résultats détaillés de chaque itération sont dans l'historique git de `eval/results/`.

## Tableau de bord

| # | Changement | Index | Hit@5 | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|---|---|
| 1 | Référence | `no_chunk` | 87 % | 0,67 | 0,62 | 0,48 | 0,58 |
| 1 | Référence | `chunk_1000` | 93 % | 0,76 | 0,56 | 0,46 | 0,57 |

## 1. Référence

**Configuration** : `ministral-14b`, top 5, filtre des événements terminés, date du jour et week-end dans le prompt.

**Constats**
- **Questions « ce week-end »** : `temp-01` échoue sur les deux index. La similarité ignore les dates et ramène des concerts de novembre à avril.
- **Effet de bord du prompt** : 8/20 (`no_chunk`) et 10/20 (`chunk_1000`) réponses restreignent au week-end des questions qui n'en parlent pas (« aucun concert de Noël ce week-end »). Le juge les classe comme évasives, d'où des *answer relevancy* à 0.
- **Découpage** : 1 687 chunks sur 6 216 ne contiennent pas la ligne `Conditions` de leur événement. Exemple : `fact-04` répond « je ne sais pas » au lieu de « sans réservation ».
- **Juge** : 2 scores manquants (JSON mal formé par `ministral-8b`), exclus des moyennes.
