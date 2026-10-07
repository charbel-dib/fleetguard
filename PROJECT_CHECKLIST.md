# FleetGuard — checklist jusqu'au portfolio

Dernière mise à jour : incrément 2. Les cases décrivent des livrables vérifiables.
Une implémentation validée ici et un push confirmé sur ton compte sont deux étapes distinctes.

- [x] **1. Données et baselines** : acquisition Scania, contrôles, split reproductible,
  pipelines logistiques, métriques de coût, inférence batch, CI et premier push confirmé.
- [x] **2. Comparaison initiale** : CV sur le développement, rôles séparés pour calibration
  et seuil, arbres/boosting, ablations initiales, diagnostic des erreurs et figures.
- [ ] **2b. Intégration GitHub de cet update** : appliquer les changements, vérifier localement,
  pousser `feat/model-comparison`, obtenir une CI verte, puis fusionner la PR vers `main`.
- [ ] **3. Optimisation et couverture ML** : recherche Optuna à budget fixé, suivi MLflow,
  ablations complémentaires, rééchantillonnage uniquement dans les folds, MLP PyTorch,
  courbes d'apprentissage et stabilité sur plusieurs seeds.
- [ ] **4. Interprétation et décision finale** : importance/permutation ou SHAP selon le modèle,
  analyse FP/FN, calibration mesurée, contraintes d'inspection et de latence, modèle figé,
  évaluation unique sur le test officiel et model card.
- [ ] **5. Service d'inférence** : FastAPI/Pydantic, health/readiness, prédiction unitaire et batch,
  limites d'entrée, version du modèle, tests de contrat et image Docker.
- [ ] **6. Interface professionnelle** : React/TypeScript, import CSV, revue des alertes,
  diagnostics de données, comparaison des expériences, accessibilité et tests de parcours.
- [ ] **7. Cloud et CI/CD** : artefacts versionnés, stockage, infrastructure déclarée,
  staging, promotion du modèle, déploiement frontend/backend et rollback vérifié.
- [ ] **8. Exploitation** : logs structurés, métriques, tests de charge, erreurs et latence,
  contrôles de dérive, budget d'hébergement et procédure de maintenance.
- [ ] **9. Portfolio Vercel** : démonstration utilisable, captures ou vidéo, résultats et limites,
  architecture, liens GitHub et app, README final et instructions reproductibles.

## Règle pour chaque prochain update

1. Livrer uniquement les fichiers ajoutés ou modifiés depuis le dernier incrément fourni.
2. Mettre à jour cette checklist, en distinguant code testé et push/CI confirmés par toi.
3. Créer une branche fonctionnelle sur un `main` à jour avant d'appliquer l'archive.
4. Exécuter les contrôles indiqués et examiner le diff avant le commit.
5. Pousser la branche, vérifier la CI et fusionner la PR ; revenir ensuite sur `main`.

Les fichiers de données, les environnements et les modèles produits localement restent ignorés.
Les petites synthèses et figures d'expériences réellement exécutées peuvent être versionnées.

## Critère « prêt pour le portfolio »

Toutes les étapes 1 à 9 sont vérifiées. La démonstration doit permettre à un visiteur d'essayer
un exemple, comprendre la décision et consulter les expériences. Les performances publiées
doivent indiquer leur partition, le modèle, le seuil et les limites du dataset.

Le modèle exact du GPU et sa VRAM seront vérifiés avant de configurer le MLP. Le benchmark
de cet incrément fonctionne sur CPU ; aucune installation CUDA n'est nécessaire pour lui.
