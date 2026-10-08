# FleetGuard — checklist jusqu'au portfolio

Dernière mise à jour : incrément 6. Les étapes précédentes, dont l'update 05, sont déclarées
terminées par toi.
Les expériences/contrôles de référence sont vérifiés ici ; une CI GitHub distante n'est pas
observée directement. Le guide du nouvel update part d'un main propre, PR précédente intégrée.

- [x] **1. Données et baselines** : snapshot Scania, contrôles, split groupé, logistiques,
  coût du challenge, inférence batch, CI et premier push confirmé.
- [x] **2. Comparaison initiale** : rôles fit/calibration/seuil/score séparés, arbres/boosting,
  ablations initiales, diagnostics et résultats mesurés.
- [x] **2b. Environnement Windows** : venv/cache hors OneDrive et mode copy, reprise confirmée.
- [x] **3. Optimisation et couverture ML** : Optuna à budget fixé, MLflow SQLite local,
  ablations, oversampling dans fit, MLP PyTorch avec arrêt interne,
  courbes par taille/epoch et stabilité des partitions sur trois seeds.
- [x] **3a. Correctif Windows du smoke de recherche** : worker séparé, nettoyage SQLite
  après sa fermeture, test de régression ; incrément déclaré terminé.
- [x] **4. Interprétation et décision finale — référence** : calibration du champion XGBoost
  sur rôle réservé, analyse FP/FN et missingness, permutation simple/groupée, budgets
  d'inspection, benchmark CPU, freeze vérifié, évaluation officielle unique et model card.
- [x] **4b. Intégration de l'update 04** : étape déclarée terminée ; la CI distante reste
  non observée directement ici.
- [x] **5. Service d'inférence — implémentation/référence** : FastAPI/Pydantic,
  live/ready, prédiction unitaire/batch, identité/hashes, schéma strict, limites de corps/lignes,
  chargement figé et tests HTTP/local sans entraînement. Bundle minimal et Docker/Compose livrés.
- [x] **5b. Intégration de l'update 05** : étape déclarée terminée ; les résultats de CI et
  de Docker sur ton compte ne sont pas observés directement ici.
- [x] **6. Interface professionnelle — implémentation/référence** : React/TypeScript,
  import CSV strict, lots bornés/progression/annulation, identité figée, revue/tri/filtres/détails,
  annotations et export ; diagnostics/expériences de référence clairement séparés du lot.
  Build, unités, parcours HTTP réel, clavier/axe et affichage mobile vérifiés localement.
- [ ] **6b. Intégration de cet update** : contrôles Windows/Python/frontend verts, parcours
  avec ta release, commit/push `feat/alert-review-ui`, PR, CI Linux/Windows/container/frontend
  verte et fusion vers main. La build reste locale jusqu'au déploiement.
- [ ] **7. Cloud et CI/CD** : artefacts et stockage versionnés, infrastructure déclarée,
  staging, promotion du modèle, frontend/backend déployés et rollback vérifié.
- [ ] **8. Exploitation** : logs/métriques, charge, erreurs/latence, dérive,
  budget d'hébergement et maintenance.
- [ ] **9. Portfolio Vercel** : démonstration utilisable, captures/vidéo, résultats/limites,
  architecture, liens GitHub/app, README final et reproduction.

## Règle de livraison et GitHub

1. Livrer uniquement les fichiers ajoutés/modifiés depuis le dernier incrément fourni.
2. Inclure cette checklist globale dans chaque update.
3. Partir d'un main propre et à jour, PR précédente intégrée, puis créer une branche fonctionnelle.
4. Copier le delta, installer, vérifier et exécuter le protocole prévu avant le commit.
5. Examiner le diff indexé, pousser la branche, vérifier la CI et fusionner ; revenir sur main.

Données, modèles, environnements et bases/reçus restent locaux et ignorés. Les synthèses et figures
réellement mesurées sont versionnées. Les futures étapes servent/déploient la release figée ;
le test officiel désormais évalué ne doit pas devenir un outil d'ajustement du modèle.
Le reçu partagé bloque les répétitions accidentelles dans la même copie du snapshot.

## Critère « prêt pour le portfolio »

Toutes les étapes sont vérifiées. Un visiteur peut essayer un exemple, comprendre la décision
et consulter des expériences traçables. Les performances publiées indiquent partition, modèle,
seuil, hypothèses et limites. La présentation distingue diagnostic APS historique et prévision
future de panne. Le transfert à une autre flotte et les décisions de maintenance réelle demandent
une validation externe.

Le modèle exact/VRAM/driver du GPU restent à vérifier avant une configuration CUDA.
Tout le protocole livré fonctionne sur CPU ; le service ne s'entraîne pas au démarrage.
