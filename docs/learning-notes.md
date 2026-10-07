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
