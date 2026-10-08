# Comprendre ce premier incrément

Ces notes relient les notions ML au code réellement exécuté. Les noms des variables étant
anonymisés, elles ne prétendent pas expliquer le mécanisme physique d'une panne.

## Accuracy et déséquilibre

Dans le train officiel, 59 000 lignes sont négatives et 1 000 positives. Toujours prédire `neg`
donnerait une accuracy de 98,33 % sur ce train, mais manquerait chaque panne APS.
Avec le coût du challenge, cela représenterait 500 000 unités. C'est un calcul sur les effectifs
du train, pas un résultat de validation ou de test.

Le rappel répond à « quelle fraction des positifs détecte-t-on ? ». La précision répond à
« quelle fraction des alertes correspond à un positif ? ». Le seuil peut augmenter le rappel
au prix d'inspections supplémentaires. `metrics.py` permet de mesurer ce compromis explicitement.

## Prétraitement et `Pipeline`

`fit` apprend un état ; `transform` applique cet état. L'imputeur apprend les médianes et la liste
des colonnes qui nécessitent un indicateur. Le scaler apprend moyenne et écart-type. Le classifieur
apprend ses coefficients. `Pipeline.fit` exécute ces étapes sur le train, puis `predict_proba`
transforme de nouvelles lignes avec les mêmes paramètres.

Une médiane calculée avant le split apprendrait indirectement la distribution de validation.
Le test `test_preprocessing_only_learns_from_training_rows` vérifie que des valeurs extrêmes
injectées après `fit` ne changent ni les médianes ni les moyennes du scaler.

L'indicateur de manque distingue une mesure imputée d'une mesure observée identique à la médiane.
Une colonne entièrement vide au train n'a pas de médiane : elle est conservée et remplie par zéro.
La normalisation ne corrige pas les outliers ; elle change seulement l'échelle. Les comparaisons
avec `log1p`, clipping appris sur train et scaling robuste seront des expériences ultérieures.

## Régression logistique

Pour des variables transformées `z`, le modèle calcule :

\[
s = w^T z + b, \qquad p = \sigma(s) = \frac{1}{1+e^{-s}}.
\]

La log-loss binaire pénalise les scores incompatibles avec le label :

\[
\ell(y,p) = -y\log(p)-(1-y)\log(1-p).
\]

Pour une observation non pondérée, le gradient de la loss par rapport à `w` est `(p-y)z`,
avant contribution de la régularisation. Les coefficients sont appris par optimisation ; le modèle
ne mémorise pas simplement une liste de seuils par capteur.

L2 pénalise les coefficients trop grands. Dans scikit-learn, `C` règle l'inverse de la force
de régularisation : diminuer `C` augmente cette contrainte. La formulation exacte de l'objectif
et le traitement de l'intercept dépendent du solver ; avec `liblinear`, l'intercept utilise une
variable synthétique et peut être régularisé. Le benchmark laisse ce comportement par défaut.

## Pondération et seuil : deux mécanismes

Avec `class_weight='balanced'`, scikit-learn calcule sur le train :

\[
\alpha_c = \frac{N}{2 N_c}.
\]

La pondération change la loss pendant l'optimisation, donc les coefficients. Le seuil change
la décision après l'entraînement. On peut pondérer et sélectionner un seuil, mais leurs effets
ne sont pas équivalents. Le rapport compare les deux régressions au seuil 0,5 et à leur propre
seuil choisi en validation.

La sortie sigmoïde d'un modèle pondéré ne représente pas automatiquement la probabilité de panne
dans la population originale. Les résultats sont appelés `positive_score`. La calibration devra
utiliser des données séparées ou des prédictions hors fold et être mesurée sur un holdout.

## Coût et recherche du seuil

La matrice de confusion a l'ordre `[[TN, FP], [FN, TP]]`. Inverser les labels ou les deux coûts
changerait complètement le modèle préféré. Les tests vérifient notamment qu'un faux positif
et un faux négatif produisent `10 + 500 = 510` unités.

Un seuil n'a d'effet sur les décisions que lorsqu'il franchit un score observé. Au lieu d'essayer
arbitrairement 100 valeurs de seuil, `threshold_curve` trie les scores, cumule les labels et évalue
chaque règle distincte. Les observations de score identique doivent être traitées ensemble :
aucun seuil ne permet d'alerter sur une seule d'entre elles.

Pour une probabilité parfaitement calibrée, les deux actions auraient les coûts attendus
`10(1-p)` et `500p`, ce qui donnerait un seuil théorique `10 / 510`. Ce résultat dépend de la
calibration et des hypothèses de coût. On ne l'applique pas directement aux scores pondérés :
le seuil de ce benchmark est choisi par le coût empirique sur validation.

## Sauvegarde et inférence

Le modèle livré comprend les transformations, les coefficients, le schéma de colonnes et le seuil.
Sauvegarder seulement le classifieur casserait les prédictions si le service recalculait d'autres
médianes ou changeait l'ordre des capteurs. `artifacts.py` contrôle le schéma, réordonne les colonnes
et recharge le pipeline complet. Un test compare les scores avant et après sérialisation.

La colonne `1` des sorties d'un classifieur n'est pas présumée positive sans contrôle :
`positive_scores` cherche explicitement la classe `1` dans `classes_`.

## Questions à examiner sur ton propre run

1. Quel est le coût de la baseline, et combien de positifs manque-t-elle ?
2. Quel modèle a la meilleure average precision ? Est-ce celui de coût minimal ?
3. La pondération améliore-t-elle le rappel à seuil 0,5 ? Quel est son coût en FP ?
4. Combien d'alertes représentent les seuils sélectionnés ?
5. Les erreurs se concentrent-elles sur les lignes ayant beaucoup de valeurs manquantes ?
6. Les coefficients seront-ils stables si on change le seed ou la régularisation ?
7. Quelles conclusions exigent encore une validation croisée ou le test final ?

Les fichiers `validation_predictions.csv` contiennent les labels, scores, décisions et fractions
de valeurs manquantes nécessaires à cette première analyse. Les réponses devront venir du run,
pas d'une description générique de l'algorithme.

## Incrément 2 : arbres, boosting et calibration

Une Random Forest agrège des arbres ajustés sur des échantillons bootstrap et des sous-ensembles
de variables. L'agrégation limite la variance des arbres individuels. `max_depth` et
`min_samples_leaf` contraignent leur complexité ; `balanced_subsample` recalcule les poids de
classes dans chaque échantillon bootstrap. Les arbres n'ont pas besoin de variables standardisées.

Le gradient boosting construit les arbres successivement pour corriger la loss du modèle courant.
Avec une loss logistique, les gradients dépendent de l'écart entre score probabiliste et label.
XGBoost utilise gradients et courbure de la loss pour construire et régulariser ses arbres.
Ces arbres sont additionnés avec un learning rate ; leur rôle diffère du vote d'une forêt.

HGB et XGBoost en mode `hist` discrétisent les variables en bins pour rechercher leurs splits.
Les valeurs manquantes peuvent suivre une branche apprise, tandis que la forêt de ce benchmark
reçoit des valeurs imputées et leurs indicateurs. Comparer les méthodes exige de préciser aussi
ces différences de prétraitement.

La variante `logistic_no_indicator` retire seulement les indicateurs de manque. La variante
`logistic_log_robust` combine compression logarithmique signée et scaling robuste. Elle peut
indiquer l'intérêt de cette combinaison, mais n'isole pas l'effet de chaque transformation.

La calibration sigmoïde apprend une transformation des sorties d'un modèle déjà entraîné.
`FrozenEstimator` empêche de réajuster les arbres HGB pendant cette étape. Une transformation
strictement croissante conserve le classement des observations et donc leur average precision,
tout en changeant les valeurs de score et potentiellement le Brier score. Elle n'améliore pas
automatiquement la décision à seuil optimisé.

Dans cet incrément, quatre usages des données sont séparés : ajustement du modèle, calibration,
sélection du seuil et mesure des performances. Une bonne loss sur le fitting ou un faible coût
sur le réglage du seuil ne suffisent pas ; `cv_fold_metrics.csv` mesure les décisions sur le scoring.

Lis `docs/comparison-methodology.md`, puis compare les résultats HGB avant/après calibration,
la régression avec/sans indicateurs et les erreurs par manque de mesures. Pour expliquer une
différence, indique quelles observations et quels paramètres ont été utilisés, avant de donner
une interprétation générale de l'algorithme.


## Incrément 3 — ce que la recherche adaptative change

Un seuil réglé hors du scoring protège l'évaluation de ce seuil pour un candidat fixé. Quand
Optuna réutilise les folds pour proposer et sélectionner des hyperparamètres, leurs scores ne
restent pas indépendants de la sélection. C'est pourquoi le meilleur coût CV n'est pas présenté
comme une estimation finale. Nested CV et test officiel sont des outils distincts, avec des coûts
et des usages différents ; cette recherche bornée n'est pas une nested CV.

Le MLP choisit les epochs dans fit, puis refit tout fit avant calibration. Choisir les epochs
sur calibration et calibrer ensuite sur les mêmes labels aurait créé une dépendance évitable.
BCEWithLogitsLoss combine sigmoid et BCE de manière stable ; la pondération des positifs change
l'objectif et les scores bruts, d'où l'intérêt de mesurer une calibration séparée. Dropout est
actif en `train()` et désactivé en `eval()` ; `no_grad()` évite les graphes durant l'inférence.

Les comparaisons `logistic_robust` et `logistic_log` isolent un changement chacune. Le variant
`logistic_log_robust` précédent combinait deux changements ; il ne permettait pas d'attribuer un
gain à un seul. L'oversampling par duplication n'est pas SMOTE ; il peut modifier les poids
implicites et la calibration. Il reste dans fit, après prétraitement appris sur les lignes originales.

Une courbe par epoch mesure la progression d'un entraînement ; une courbe par taille de données
mesure l'effet de disposer de plus de lignes d'entraînement à paramètres fixes. Les deux sont
livrées, avec des axes et rôles explicitement différents. Une sensibilité aux partitions ne
mesure pas toutes les initialisations aléatoires du réseau.

MLflow suit les essais et leurs artefacts ; Optuna décide les paramètres à essayer. Un run
`FINISHED` dans MLflow ne prouve pas à lui seul que le protocole est valide : les manifestes,
les tests de non-interférence des labels et la documentation restent nécessaires.

## Update 04 — from scores to a frozen decision

- A calibration map can improve score scaling without changing ranking or alerts. On this release,
  validation Brier fell from 0.013906 to 0.007209, while FP/FN remained 410/11.
- Calibration selection uses grouped OOF within the assigned calibration role. Hyperparameter
  search had already seen these development rows in CV, so the official test is the final benchmark.
- The calibrated empirical threshold (0.002490) differs from the raw threshold (0.176826). Changing
  the numeric scale does not imply the policy became more aggressive; compare actual decisions.
- Inspection budgets are fitted on threshold-role data. The 3% scenario flagged 2.92% on validation
  and recalled 87% of positives. A fixed threshold cannot guarantee 3% on a shifted future batch.
- Permutation measures sensitivity, not cause. Column aa_000 had the largest measured increase in
  fixed-threshold cost, while ck_000 affected AP more strongly. Correlated blocks can hide importance.
- The missingness slices matter: 8/11 validation FN occurred below 5% missingness. Among 84 rows
  above 50% missingness, there were 24 FP and **zero positives**; no APS-recall conclusion follows.
- A 33,497-byte model file does not imply a 33-KB inference process. Whole-process RSS includes
  the interpreter, dataframe/scientific libraries and native buffers. HTTP latency is not measured yet.
- After freeze, official test gave 475 FP, 16 FN, recall 95.73%, precision 43.05%, AP 0.88444 and
  challenge cost 12,750 on 16,000 rows (375 positives). No model or threshold change followed.
- Bootstrap intervals are conditional on the fixed model/snapshot. Complete duplicate-feature
  groups are resampled together, but unavailable truck IDs prevent modeling vehicle dependence.
- Once test outcomes are public, subsequent model iteration cannot present that same test as fresh
  unseen evidence. The next increments build serving, UI and deployment around this frozen release.

Read [release methodology](release-methodology.md), [ADR 0004](decisions/0004-freeze-calibration-policy-and-consume-test-once.md)
and the [model card](results/release-04.md).


## Incrément 5 — rendre la décision disponible en HTTP

Le pipeline appris et le seuil forment une seule décision. L'API recharge cet état une fois au
startup, vérifie son freeze et réalise une prédiction de chauffe. Elle ne recalibre pas à partir
des données entrantes. Le service devient ready seulement après ces contrôles ; live indique
que l'application répond. Un modèle invalide fait échouer le démarrage.

Un schéma typé ne remplace pas le contrat ML : Pydantic vérifie les types/bornes, et le predictor
vérifie que chaque capteur attendu est présent. `null` décrit une mesure absente ; omettre une clé
rend le schéma incomplet. Refuser les booléens, chaînes numériques et clés JSON dupliquées évite
les interprétations silencieuses. La limite de corps compte les octets reçus plutôt que de croire
un Content-Length fourni par le client.

Une fonction async ne rend pas un calcul scientifique non bloquant. L'inférence synchrone passe
dans un thread, avec un seul slot et un lock ; la boucle HTTP reste disponible pour un health
admis pendant ce calcul. La limite serveur borne les connexions/tasks et peut produire 503 en
saturation. Elle ne prouve pas une capacité de charge : celle-ci devra être mesurée.

Le bundle contient le contrat de la release ; l'image contient le code et les dépendances. Les
séparer permet de tester le container avec une fixture synthétique et de monter le modèle figé
en lecture seule. L'identité/hashes dans chaque réponse permettent de vérifier quel modèle répond.
Les checksums ne rendent pas un joblib inconnu sûr : la confiance et la promotion de l'artefact
restent des responsabilités de déploiement.

L'égalité des scores HTTP/local sur quatre lignes du train vérifie l'intégration. Elle ne mesure
ni la généralisation ni la latence en charge. Le test officiel a déjà été évalué dans l'update 04 ;
l'API réutilise le freeze et ne réouvre pas cette expérience pour ajuster la décision.

Lis [le contrat API](api-contract.md) et [ADR 0005](decisions/0005-load-one-frozen-release-and-bound-http-inputs.md).


## Incrément 6 — observer un lot sans inventer son évaluation

Le type TypeScript décrit ce que le code attend ; il ne valide pas une réponse HTTP au runtime.
Le client vérifie aussi schéma, identité, ordre, bornes des scores et cohérence score/seuil/label.
Une identité modifiée au milieu des lots invalide le résultat complet, même si chaque HTTP vaut 200.
La limite par octets se calcule sur le JSON UTF-8 envoyé, pas sur le poids du CSV ni sa seule longueur.

Une observation absente est null ; une colonne absente rend le contrat incomplet. La validation
CSV précède l'envoi, puis l'API garde ses propres contrôles. Les lignes entièrement vides en capteurs
restent des observations : les perdre silencieusement changerait l'index de la revue.

Un lot sans labels peut mesurer sa fraction d'alertes et son manque de mesures, mais pas son rappel,
sa précision ou ses FN. La vue des preuves de référence nomme une autre expérience et son snapshot ;
elles ne deviennent pas des performances du modèle actif. Pos/neg restent des attributions APS/hors
APS, pas des horizons de panne ni des garanties de santé.

Les tests de composants seuls n'auraient pas vérifié le proxy/origin, les assets de production et la
vraie API. Les parcours Playwright utilisent la build et un serveur synthétique réel, avec de petits
lots forçant plusieurs requêtes. Le dialogue natif gère focus/Escape ; axe et le contrôle mobile
révèlent des problèmes concrets de contraste/débordement, sans constituer une certification WCAG.

La build sépare code et agrégats versionnés des données/model locaux. La suite déploiera cette
interface et la release avec une politique explicite d'accès, de promotion et de rollback.


## Increment 07: immutable cloud promotion

A model release, a code commit and an OCI image digest are different identities. A deployment
candidate binds all three; an ECR tag alone is insufficient. S3 object version IDs pin the exact
trusted model archive. The archive's SHA256 checks integrity, while freeze/model contracts check
semantic consistency. Checksums do not make unknown joblib files trustworthy.

OIDC replaces stored GitHub AWS keys; exact observed environment subjects restrict assumption.
The execution role can pull images/write logs while the application task needs no AWS permission.
A stable ECS service may reflect automatic rollback to the old task, so the requested PRIMARY
revision must also match. Health/API/model/code/asset checks and verified restoration supply
operational evidence; mocked AWS tests verify orchestration but do not establish cloud success.

Terraform owns infrastructure, CI owns later image revisions. Separate state keys and data paths
avoid crossing staging/production. Budgets alert rather than cap spending. The delivered AWS
configuration is pending real CI/provider validation and cloud acceptance before portfolio claims.
