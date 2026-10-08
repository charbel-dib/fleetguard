# ADR 0003 — recherche bornée et arrêt MLP dans fit

Statut : accepté pour l'incrément 3.

## Contexte

La comparaison de l'incrément 2 a séparé fit, calibration, seuil et scoring. Une recherche
adaptative et un MLP introduisent deux nouveaux choix : hyperparamètres et nombre d'epochs.
Consommer le rôle de calibration pour choisir les epochs puis calibrer le même modèle affaiblirait
l'indépendance du calibrateur ; choisir les epochs sur threshold ou score contaminerait leurs rôles.

## Décision

TPE séquentiel, seeds et budgets fixes, premier essai de référence inclus dans le budget.
Les scores CV sont des scores de sélection, explicitement non indépendants après recherche.
Un split groupé interne à fit choisit les epochs ; tout fit est ensuite refitté avec des poids
réinitialisés et ce nombre d'epochs. Le calibrateur utilise uniquement le rôle séparé de calibration.

Le champion est écrit avant diagnostics et validation. Les diagnostics à paramètres fixés ne
changent pas la sélection. Le test officiel reste intact jusqu'au gel expérimental.
Les contrôles logistiques partagent un budget de 100 itérations. Une non-convergence les exclut
explicitement du classement, sans artefact publié. Les erreurs imprévues et interruptions font
échouer le run entier.
MLflow utilise une base SQLite locale et sa télémétrie est désactivée.
Le lockfile initial fournissait PyTorch CPU. Après vérification de la GTX 1650 Ti, il fournit
PyTorch 2.8.0 avec CUDA 12.6 sur Linux/Windows et le paquet XGBoost complet. Les configurations
fournies sélectionnent CUDA ; les configurations de smoke restent sur CPU.

## Conséquences

Le coût d'entraînement du MLP inclut sélection et refit. Le budget limité est exploratoire.
Les scores de recherche sont optimistes ; les partitions retenues ne sont pas un nouveau test
aveugle. La sensibilité mesurée concerne les seeds de partition avec seed modèle constante.
La reprise automatique d'études et un registre distant restent hors de cet incrément.
