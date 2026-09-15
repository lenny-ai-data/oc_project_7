# Itérations d'évaluation

Jeu de test : `eval/test_set.json` (20 questions, date de référence 15/09/2026).
Hit@5 sur les 15 questions qui attendent des événements ; scores Ragas (juge `ministral-8b`) sur ces mêmes 15 questions.
Les résultats détaillés de chaque itération sont dans l'historique git de `eval/results/`.

## Tableau de bord

| # | Changement | Index | Hit@5 | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|---|---|
| 1 | Référence | `no_chunk` | 87 % | 0,67 | 0,62 | 0,48 | 0,58 |
| 1 | Référence | `chunk_1000` | 93 % | 0,76 | 0,56 | 0,46 | 0,57 |
| 2 | Prompt : week-end seulement si demandé | `no_chunk` | 87 % | **0,84** | **0,75** | 0,51 | 0,67 |
| 2 | Prompt : week-end seulement si demandé | `chunk_1000` | 93 % | 0,78 | **0,74** | 0,45 | 0,57 |
| 3 | Découpage : conditions dans chaque chunk | `chunk_1000` | 93 % | **0,86** | **0,80** | 0,53 | 0,65 |

## 1. Référence

**Configuration** : `ministral-14b`, top 5, filtre des événements terminés, date du jour et week-end dans le prompt.

**Constats**
- **Questions « ce week-end »** : `temp-01` échoue sur les deux index. La similarité ignore les dates et ramène des concerts de novembre à avril.
- **Effet de bord du prompt** : 8/20 (`no_chunk`) et 10/20 (`chunk_1000`) réponses restreignent au week-end des questions qui n'en parlent pas (« aucun concert de Noël ce week-end »). Le juge les classe comme évasives, d'où des *answer relevancy* à 0.
- **Découpage** : 1 687 chunks sur 6 216 ne contiennent pas la ligne `Conditions` de leur événement. Exemple : `fact-04` répond « je ne sais pas » au lieu de « sans réservation ».
- **Juge** : 2 scores manquants (JSON mal formé par `ministral-8b`), exclus des moyennes.

## 2. Prompt : dates du week-end utilisées seulement si la question en parle

**Changement** (`rag/chain.py`) : « ce week-end désigne le … » devient « Si la question parle du week-end, il s'agit du … ; sinon, ne limite pas ta réponse à une période. »

**Constats**
- **Effet de bord fortement réduit** : les réponses qui parlent du week-end sans que la question le demande passent de 8 à 3 (`no_chunk`) et de 10 à 5 (`chunk_1000`). Ce comptage est automatique et majore le problème : `fact-05`, par exemple, cite à juste titre le 19-20 septembre, qui est la date de l'exposition.
- **Réponses** : *faithfulness* +0,17 et *answer relevancy* +0,13 pour `no_chunk`, *answer relevancy* +0,18 pour `chunk_1000`. Les concerts de Noël, les films espagnols et la sortie avec un bébé sont maintenant proposés.
- **Recherche inchangée** : le hit@5 reste identique (87 % et 93 %, mêmes échecs), ce qui est attendu puisque seul le prompt a changé.
- **Bruit du juge** : *context precision* et *context recall* ne dépendent que des contextes et de la référence, qui n'ont pas changé. Leurs moyennes bougent pourtant (*context recall* de `no_chunk` : 0,58 → 0,67 ; `info-01` passe de 0 à 1). Avec `ministral-8b`, **un écart de moins de 0,1 n'est pas significatif** : seules les tendances nettes comptent.
- **Limites restantes** :
  - `temp-01` et `temp-02` (`no_chunk`) échouent dès la recherche : le système répond honnêtement « rien ce week-end », sans aider l'utilisateur. Côté `chunk_1000`, `temp-01` présente une visite de campus comme un « atelier-concert ».
  - `info-01` : un « je ne connais pas le tarif » correct obtient une *answer relevancy* de 0, car Ragas note 0 toute réponse jugée évasive.
  - `fact-04` (`chunk_1000`) : les conditions sont toujours absentes du chunk retenu (itération 3).
  - `hors-01` et `hors-02` : refus, suivis d'un recadrage vers les événements du week-end, cohérent pour un assistant de recommandation.

## 3. Découpage : conditions d'accès répétées dans chaque chunk

**Changement** (`rag/documents.py`) : la ligne `Conditions` (tarif, réservation), en fin de texte, est déplacée dans l'en-tête répété de chaque chunk. L'index `chunk_1000` est reconstruit (6 273 chunks au lieu de 6 216) ; `no_chunk` n'est pas concerné.

**Constats**
- **Cause corrigée** : aucun chunk n'est plus privé des conditions de son événement (1 687 auparavant).
- **`fact-04` résolu** : « Non, il n'est pas nécessaire de réserver… gratuite et sans réservation, limitée à 80 personnes ». Ses scores passent de 0,40 / 0 / 0 / 0,50 à 0,83 / 0,86 / 1,0 / 1,0.
- **Moyennes** : *faithfulness* 0,78 → 0,86, *answer relevancy* 0,74 → 0,80, *context precision* 0,45 → 0,53, *context recall* 0,57 → 0,65. L'amélioration est générale ; hors `fact-04`, chaque écart reste dans le bruit du juge. Exemple de bruit : `fact-01` a les mêmes sources, mais son *context recall* passe de 0,5 à 1,0.
- **Recherche** : le hit@5 reste à 93 %, mais les sources changent pour 10 questions sur 15. Les en-têtes plus riches modifient les vecteurs des chunks.
- **Limites restantes** :
  - `temp-01` (« concerts gratuits ce week-end ») échoue toujours : la recherche ignore les dates.
  - *Context precision* et *context recall* restent à 0 sur les questions larges (`reco-01`, `temp-01` à `temp-03`) : la référence cite quelques exemples, et les événements retrouvés, parfois tout aussi valables, n'y figurent pas. C'est une limite de l'annotation par exemples pour les questions ouvertes.
  - `info-01` : réponse correcte (« je ne dispose pas du tarif »), mais *answer relevancy* reste à 0 (réponse jugée évasive), et le modèle suggère encore de consulter le site du Château d'Eau.

## Bilan provisoire

| Index | Itération | Hit@5 | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|---|
| `no_chunk` | 2 | 87 % | 0,84 | 0,75 | 0,51 | 0,67 |
| `chunk_1000` | 3 | **93 %** | **0,86** | **0,80** | **0,53** | 0,65 |

`chunk_1000` devance légèrement `no_chunk`. Le seul écart net est le hit@5 (une question de plus retrouvée, `temp-02`) ; les écarts Ragas restent dans le bruit du juge.
