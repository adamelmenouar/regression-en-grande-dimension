# Insurance Fraud Detection: Logistic Regression vs SVM vs Random Forest vs Voting

Projet réalisé dans le cadre du cours **Machine Learning – INSEA S5**  
Encadré par **M. Hicham Janati**

---

## Objectif du projet

Ce projet vise à construire un pipeline complet de **détection de fraude** dans les sinistres d’assurance à l’aide de techniques de **Machine Learning**.

L’objectif principal est de :
- **Prédire** si un sinistre est **frauduleux** (`fraud_reported = Y`) ou **non frauduleux** (`fraud_reported = N`)
- **Comparer** plusieurs modèles de classification :
  - **Logistic Regression**
  - **SVM**
  - **Random Forest**
  - **Voting Classifier**
- **Évaluer** les performances avec plusieurs métriques (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Interpréter** le modèle via **SHAP** (sur Random Forest)
- **Choisir** un modèle final de manière rigoureuse grâce à la **validation croisée**

---

## Données utilisées

- Fichier : `insurance_claims.csv`
- Taille : ~1000 observations (selon le nettoyage)
- Variable cible : `fraud_reported` (`Y` = fraude, `N` = non fraude)
- Données anonymisées (identifiants masqués)

---

## Méthodologie

1. **Chargement & Nettoyage**
   - Gestion des valeurs manquantes (`na_values='?'`)
   - Suppression de colonnes non pertinentes / redondantes
   - Remplacement de certaines modalités manquantes par `Unknown`

2. **Analyse exploratoire (EDA)**
   - Distribution de la cible (déséquilibre fraude/non-fraude)
   - Analyse des variables numériques et catégorielles
   - Corrélations et visualisations
   - Outliers (ex : `umbrella_limit` avec capping au 99e centile)

3. **Prétraitement**
   - Suppression des dates (selon la stratégie)
   - One-Hot Encoding (`pd.get_dummies`)
   - Standardisation (`StandardScaler`)

4. **Gestion du déséquilibre**
   - Utilisation de **SMOTE** sur l’échantillon d’entraînement uniquement

5. **Modélisation**
   - Entraînement et comparaison des 4 modèles
   - Évaluation sur un jeu de test (split stratifié)

6. **Validation croisée multi-métriques**
   - `StratifiedKFold` (5 folds)
   - `cross_validate` avec :
     - Accuracy, Precision, Recall, F1, ROC-AUC
   - Choix du meilleur modèle (dans ce projet : **Random Forest**)

7. **Évaluation finale & ROC**
   - Matrice de confusion
   - Courbes ROC pour tous les modèles
   - ROC-AUC

8. **Interprétation**
   - **SHAP TreeExplainer** appliqué à Random Forest
   - Importance globale (bar plot), beeswarm, dependence plots

---

## Structure du projet

regression-en-grande-dimension/
│
├── data/  Dataset (à ajouter localement)
│   └── insurance_claims.csv
│
├── notebooks/  Analyses et modèles
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_preprocessing_smote.ipynb
│   ├── 03_modeling_evaluation.ipynb
│   └── 04_shap_interpretability.ipynb
│
├── src/  Scripts Python réutilisables (optionnel mais propre)
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── shap_utils.py
│
├── figures/  Graphiques générés (EDA, ROC, confusion, SHAP, etc.)
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── roc_curves_all_models.png
│   ├── confusion_matrix_rf.png
│   ├── shap_summary_bar.png
│   └── shap_beeswarm.png
│
├── report/  Rapport et livrables PDF
│   └── Project_Report_FraudDetection.pdf
│
├── requirements.txt  Librairies nécessaires
└── README.md  Description du projet


---

## Résultats (résumé)

- Les performances ont été évaluées avec **Accuracy, Precision, Recall, F1 et ROC-AUC**
- La **validation croisée multi-métriques** a permis de choisir un modèle robuste
- **Random Forest** a obtenu les meilleurs résultats globaux en CV (F1/ROC-AUC)
- L’évaluation finale est faite sur un **test réel déséquilibré**, donc les métriques sont plus réalistes (souvent plus faibles que la CV sur SMOTE)

---

## Technologies utilisées

- **Langage** : Python  
- **Librairies principales** :
  - `numpy`, `pandas`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `imbalanced-learn` (SMOTE)
  - `shap`

---

