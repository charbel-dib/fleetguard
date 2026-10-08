# Optimisation : budget, rôles et limites

Le config `configs/optimization.toml` référence `comparison.toml`. Le split externe Scania demeure
48 000 observations de développement / 12 000 de validation conservée. Le test officiel n'est pas lu.
Les hash de source, environnement, paramètres et manifestes rendent un run inspectable.

## Rôles dans chaque fold

| Rôle | XGBoost / logistiques | MLP |
|---|---|---|
| Fit (19 200 lignes environ) | Prétraitement et apprentissage ; oversampling uniquement ici | Split groupé interne 80/20 pour choisir les epochs, puis refit de tout fit |
| Calibration (6 400) | Réservé, inutilisé | Sigmoid sur `FrozenEstimator` du MLP refitté |
| Threshold (6 400) | Recherche exacte du seuil de coût minimal | Même recherche exacte après calibration |
| Score (16 000) | Métriques des décisions figées | Même protocole |

Le split interne MLP préserve les groupes de features identiques. Pendant la sélection des epochs,
le prétraitement est appris sur le sous-fit uniquement. La perte d'arrêt est une BCE pondérée
par le ratio négatifs/positifs du sous-fit. Après sélection, prétraitement et réseau sont réinitialisés
et entraînés sur tout fit pour le nombre d'epochs retenu ; la pondération vient alors de tout fit.
Les poids de la dernière epoch du refit sont sauvegardés. Les poids du réseau sont sérialisés en
arrays NumPy, avec version PyTorch vérifiée au chargement ; l'inférence reconstruit le réseau sur CPU.

La calibration ne réentraîne pas le réseau. Son Brier score, sa courbe de fiabilité et ses limites
doivent être examinés : avoir un calibrateur ne prouve pas une bonne calibration en production.

## Recherche et contrôles

Optuna utilise TPE seed 44, deux essais de démarrage, `n_jobs=1` et aucun pruning. Le premier essai
est fixé : référence de l'update 02 pour XGBoost ; MLP 64 × 2, dropout 0,1, AdamW 0,001,
weight decay 0,0001 et batch 1024. Ces essais comptent dans les budgets 6 et 4.

| Famille | Espace de recherche |
|---|---|
| XGBoost CPU hist | Arbres 80–220 par pas de 10 ; profondeur 2–6 ; learning rate 0,03–0,2 log ; subsample et colsample 0,6–1 ; L2 0,01–20 log ; pondération none/balanced |
| MLP PyTorch | Largeur 64/128 ; couches 1/2 ; dropout 0–0,3 ; learning rate 0,0005–0,005 log ; weight decay 0,000001–0,01 log ; batch 512/1024 ; epochs ≤24, patience 5 |

La pondération XGBoost équilibrée est calculée à partir de fit uniquement. MLP utilise toujours
une BCE pondérée, une imputation médiane avec indicateurs et StandardScaler. Le réseau utilise
ReLU, Dropout, une sortie logit et `BCEWithLogitsLoss`, AdamW, `train()`/`eval()` et `no_grad()`.
Les seeds et threads sont bornés. Le RNG PyTorch et le nombre de threads sont restaurés après fit.
La reproductibilité est testée dans un environnement CPU identique ; elle n'est pas promise bit
à bit entre systèmes, versions, GPU ou partitions différentes.

Les contrôles fixes sont XGBoost référence, logistique originale, RobustScaler seul, logarithme
signé seul et RandomOverSampler seul. Le logarithme `sign(x)*log1p(abs(x))` est stateless et précède
l'imputation. L'oversampler, après prétraitement appris sur les lignes originales de fit, monte le
ratio positifs/négatifs à environ 0,1 si nécessaire ; il n'abaisse pas un ratio déjà plus élevé.
Les autres rôles gardent leur prévalence d'origine. Il s'agit d'oversampling par duplication, pas de SMOTE.

Les quatre contrôles logistiques ont un budget commun de 100 itérations (`control_max_iter`),
avec C/solver/tol identiques. Seuls les contrôles convergés sur tous les folds sont comparés.
Une non-convergence est enregistrée dans `control_status.csv` et `failure.json`, le run enfant
MLflow passe à FAILED et ce contrôle ne reçoit ni score partiel au classement ni artefact final.
La recherche et les autres contrôles continuent. Une erreur imprévue reste fatale au run.

## Ce que les scores permettent de conclure

Chaque trial score toutes les lignes du développement exactement une fois sur trois folds.
Le seuil n'est jamais ajusté sur ces labels de scoring. **Les mêmes scores choisissent néanmoins
les hyperparamètres** : le meilleur coût CV subit donc un biais de sélection. Cette recherche
n'est pas une nested CV. Le budget réduit est exploratoire, pas une preuve d'optimum global.

Le meilleur trial de chaque famille est retenu par coût/observation, puis AP moyenne, puis numéro
d'essai. Familles optimisées et contrôles sont ensuite classés par coût, AP, puis nom. Le champion
est écrit avant les diagnostics et avant de lire les labels de validation conservée pour scoring.
La validation a déjà été inspectée dans les incréments précédents : ce n'est pas un test aveugle.
Les métriques officielles finales restent réservées à l'étape 4 après gel expérimental.

## Stabilité et courbes d'apprentissage

Les paramètres sélectionnés restent fixes. Les seeds 43/44/45 modifient les partitions groupées
CV et les rôles internes ; la seed des modèles demeure 42. Cette mesure est une sensibilité aux
partitions, pas une étude séparée de toutes les initialisations du réseau. Aucun diagnostic ne
remplace le champion ou ne sert à retuner le seuil final.

Les courbes par taille utilisent 25 %, 50 %, 100 % du rôle fit, par sous-ensembles imbriqués de
groupes entiers, stratifiés sur le label positif maximal du groupe. Les tailles réalisées sont
enregistrées par fold. Calibration, seuil et scoring restent fixes. Les rôles des diagnostics
désignent les partitions éligibles ; les sous-ensembles effectifs sont déterministes depuis fit,
la fraction et la seed. Les audits internes MLP enregistrent les lignes réellement utilisées.
La courbe par epoch est distincte : elle montre la BCE d'arrêt à l'intérieur de fit.

Les folds/seeds corrélés ne fournissent pas des intervalles de confiance. Une meilleure valeur
sur quelques folds ne garantit pas un gain sur une nouvelle flotte.

## MLflow local

SQLite et artefacts locaux sont imposés par le client, indépendamment d'un `MLFLOW_TRACKING_URI`
externe. La télémétrie est désactivée avant import et vérifiée. Le run parent rassemble provenance,
paramètres, synthèses et figures ; chaque essai/contrôle a un run enfant avec ses métriques.
Les bases de suivi, modèles et données restent ignorés par Git. Les études Optuna sont persistées
par run mais la reprise automatique n'est pas implémentée ; une relance démarre un nouveau budget.

## Références primaires

- [Optuna : reproductibilité et recherche séquentielle](https://optuna.readthedocs.io/en/stable/faq.html)
- [MLflow : suivi SQLite](https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database/)
- [MLflow : désactivation de la télémétrie](https://mlflow.org/docs/latest/community/usage-tracking/)
- [PyTorch : installation](https://docs.pytorch.org/get-started/locally/)
- [uv : PyTorch CUDA et index explicite](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [imbalanced-learn : fuites de données](https://imbalanced-learn.org/stable/common_pitfalls.html)
