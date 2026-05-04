# Projet Data Science — Prédiction du churn client

## Auteur

Projet réalisé par **Antoine** et **François** dans le cadre d’un projet de fin d’année en data science.

## Présentation

Ce projet a été réalisé dans le cadre d’un projet de fin d’année en data science.  
L’objectif est de prédire le **churn client**, c’est-à-dire repérer les clients qui risquent de quitter le service.

L’idée est d’utiliser des données clients pour :
- détecter les profils à risque,
- aider à mettre en place des actions de rétention,
- estimer le revenu potentiellement perdu.

---

## Contenu du projet

Le projet contient :
- un script d’entraînement des modèles : `train.py`
- une API FastAPI : `api.py`
- un dashboard Streamlit : `app.py`
- un notebook d’analyse exploratoire : `traitement_eda.ipynb`
- un notebook de modélisation : `analysis_updated.ipynb`

---

## Structure du projet

```bash
Projet-data-science/
│
├── Code/
│   ├── train.py
│   ├── api.py
│   ├── app.py
│   ├── analysis_updated.ipynb
│   └── traitement_eda.ipynb
│
├── SOURCE/
│   └── dataset.csv
│
├── artifacts/
│   ├── best_model.joblib
│   ├── model_config.json
│   ├── model_metrics.csv
│   ├── test_predictions.csv
│   └── feature_importance.csv
│
├── mlflow.db
└── README.md
```

---

## Objectif

Le but du projet est de comparer plusieurs modèles de machine learning pour prédire le churn, puis de déployer le meilleur modèle dans une petite application exploitable.

---

## Variables utilisées

Pour garder une cohérence entre l’entraînement, l’API et le dashboard, le modèle final utilise 10 variables :

- `gender`
- `age`
- `tenure_months`
- `contract_type`
- `monthly_logins`
- `weekly_active_days`
- `avg_session_time`
- `monthly_fee`
- `payment_failures`
- `nps_score`

Variable cible :
- `churn`

---

## Modèles testés

Quatre modèles ont été comparés :

- Logistic Regression
- Random Forest
- Gradient Boosting
- MLPClassifier

Le MLP a été gardé pour répondre à la consigne d’utiliser un modèle de deep learning, mais les meilleurs résultats ont été obtenus avec les modèles classiques.

---

## Méthode utilisée

Le projet suit les étapes suivantes :

1. analyse exploratoire des données,
2. préparation des variables,
3. entraînement de plusieurs modèles,
4. validation croisée stratifiée,
5. optimisation du seuil de décision,
6. comparaison des performances,
7. déploiement du modèle avec une API,
8. création d’un dashboard.

Le suivi des expériences est fait avec **MLflow**.

---

## Métriques utilisées

Les modèles sont évalués avec :

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score

Le problème étant déséquilibré, la **PR-AUC** et le **Recall** sont particulièrement importants.

---

## Lancer le projet

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

Si tu n’as pas de `requirements.txt` :

```bash
pip install pandas numpy scikit-learn joblib mlflow fastapi uvicorn streamlit requests plotly matplotlib seaborn jupyter notebook
```

---

### 2. Lancer l’entraînement

Depuis le dossier `Code` :

```bash
cd Code
python train.py
```

Ce script permet de :
- charger les données,
- entraîner les modèles,
- comparer les résultats,
- choisir le meilleur modèle,
- sauvegarder les fichiers dans `artifacts/`,
- enregistrer le modèle dans MLflow.

---

### 3. Lancer l’API

Depuis le dossier `Code` :

```bash
uvicorn api:app --reload
```

L’API sera disponible ici :

```bash
http://127.0.0.1:8000
```

Documentation Swagger :

```bash
http://127.0.0.1:8000/docs
```

---

### 4. Lancer le dashboard

Depuis le dossier `Code` :

```bash
python -m streamlit run app.py
```

Le dashboard permet de :
- saisir les informations d’un client,
- obtenir une probabilité de churn,
- afficher le niveau de risque,
- comparer les modèles,
- visualiser l’importance des variables.

---

## Fichiers générés

Après l’entraînement, le dossier `artifacts/` contient :

- `best_model.joblib` : sauvegarde locale du meilleur modèle
- `model_config.json` : configuration du modèle et seuil retenu
- `model_metrics.csv` : comparaison des modèles
- `test_predictions.csv` : prédictions sur le jeu de test
- `feature_importance.csv` : importance des variables

---

## Résultats généraux

Les résultats montrent que :
- les modèles de type **Random Forest** et **Gradient Boosting** sont les plus performants,
- la **Logistic Regression** reste une bonne baseline,
- le **MLP** est moins adapté sur ce dataset,
- l’optimisation du seuil améliore la détection des churners.

---

## Notebooks

### `traitement_eda.ipynb`
Ce notebook sert à l’analyse exploratoire :
- compréhension du dataset,
- valeurs manquantes,
- distributions,
- corrélations,
- premières interprétations métier.

### `analysis_updated.ipynb`
Ce notebook sert à la modélisation :
- comparaison des modèles,
- validation croisée,
- optimisation du seuil,
- courbes ROC,
- matrices de confusion,
- importance des variables.

---

## Conclusion

Ce projet met en place une chaîne complète de data science :
- exploration des données,
- entraînement de plusieurs modèles,
- validation croisée,
- optimisation du seuil,
- suivi avec MLflow,
- déploiement avec FastAPI,
- visualisation avec Streamlit.

---

