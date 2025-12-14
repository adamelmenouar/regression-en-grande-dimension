# Détection de fraude dans les sinistres d’assurance (Machine Learning)

Ce projet vise à **détecter automatiquement les sinistres frauduleux** à partir d’un dataset d’assurance.
Le notebook couvre tout le pipeline : **EDA → nettoyage → encodage → standardisation → SMOTE → entraînement de modèles → évaluation → validation croisée → ROC → SHAP**.

---

## Contenu du dépôt

- `detection de fraude.ipynb` : notebook principal (pipeline complet)
- `insurance_claims.csv` : dataset (à placer en local dans le dossier `data/`)

---

## Dataset

- Fichier : `insurance_claims.csv`
- Variable cible : `fraud_reported` (Y = fraude, N = non fraude)

---

