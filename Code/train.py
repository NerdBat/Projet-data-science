from pathlib import Path
import json
import time
import warnings
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# =========================================================
# Configuration
# =========================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "SOURCE" / "dataset.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

TARGET = "churn"
MODEL_NAME = "Churn_Classifier"
EXPERIMENT_NAME = "churn_project"

# Même variables que l'API / dashboard
FEATURES = [
    "gender",
    "age",
    "tenure_months",
    "contract_type",
    "monthly_logins",
    "weekly_active_days",
    "avg_session_time",
    "monthly_fee",
    "payment_failures",
    "nps_score",
]

# Base MLflow locale
MLFLOW_DB = (BASE_DIR / "mlflow.db").as_posix()
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()


# =========================================================
# Fonctions utiles
# =========================================================
def choose_best_threshold(y_true, y_proba):
    """
    Choisit le meilleur seuil avec le F2-score
    (on donne plus d'importance au recall).
    """
    best_threshold = 0.5
    best_f2 = -1

    for threshold in np.arange(0.10, 0.91, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

        if f2 > best_f2:
            best_f2 = f2
            best_threshold = round(float(threshold), 2)

    return best_threshold


def build_pipeline(model, numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    return pipeline


def wait_until_ready(model_name, version, timeout=30):
    """
    Attend un peu que MLflow finisse l'enregistrement.
    """
    start = time.time()
    while time.time() - start < timeout:
        mv = client.get_model_version(model_name, version)
        if mv.status == "READY":
            return
        time.sleep(1)


# =========================================================
# Chargement des données
# =========================================================
df = pd.read_csv(DATA_PATH)

missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Colonnes manquantes : {missing_cols}")

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"Train : {X_train.shape} | Test : {X_test.shape}")
print(f"Taux de churn : {y.mean():.2%}")


# =========================================================
# Modèles à comparer
# =========================================================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE,
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE,
    ),
}

cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

results = []
trained_pipelines = {}
best_thresholds = {}

# =========================================================
# Entraînement + CV + seuil + logs MLflow
# =========================================================
for model_name, model in models.items():
    print(f"\n===== {model_name} =====")

    pipeline = build_pipeline(model, numeric_features, categorical_features)

    with mlflow.start_run(run_name=model_name):
        # Validation croisée
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        # Probabilités out-of-fold pour choisir le seuil
        oof_proba = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        best_threshold = choose_best_threshold(y_train, oof_proba)

        # Fit final sur tout le train
        pipeline.fit(X_train, y_train)

        # Test final
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= best_threshold).astype(int)

        metrics = {
            "model": model_name,
            "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
            "cv_pr_auc_mean": float(np.mean(cv_results["test_pr_auc"])),
            "cv_precision_mean": float(np.mean(cv_results["test_precision"])),
            "cv_recall_mean": float(np.mean(cv_results["test_recall"])),
            "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
            "threshold": best_threshold,
            "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
            "test_pr_auc": float(average_precision_score(y_test, y_proba)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        results.append(metrics)
        trained_pipelines[model_name] = pipeline
        best_thresholds[model_name] = best_threshold

        # Logs MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("features", ", ".join(FEATURES))
        mlflow.log_param("cv_splits", CV_SPLITS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("threshold", best_threshold)

        for key, value in metrics.items():
            if key != "model":
                mlflow.log_metric(key, value)

        print(f"CV ROC-AUC : {metrics['cv_roc_auc_mean']:.4f}")
        print(f"CV PR-AUC  : {metrics['cv_pr_auc_mean']:.4f}")
        print(f"CV Recall  : {metrics['cv_recall_mean']:.4f}")
        print(f"Seuil      : {metrics['threshold']:.2f}")
        print(f"Test F1    : {metrics['test_f1']:.4f}")


# =========================================================
# Choix du meilleur modèle
# =========================================================
results_df = pd.DataFrame(results).sort_values(
    by=["cv_pr_auc_mean", "cv_recall_mean", "cv_roc_auc_mean"],
    ascending=False,
)

best_model_name = results_df.iloc[0]["model"]
best_pipeline = trained_pipelines[best_model_name]
best_threshold = best_thresholds[best_model_name]

print("\n================================================")
print(f"Meilleur modèle retenu : {best_model_name}")
print(f"Seuil retenu : {best_threshold}")
print("================================================")
print(results_df.to_string(index=False))

# Évaluation détaillée sur test
best_y_proba = best_pipeline.predict_proba(X_test)[:, 1]
best_y_pred = (best_y_proba >= best_threshold).astype(int)

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, best_y_pred))


# =========================================================
# Sauvegarde locale
# =========================================================
joblib.dump(best_pipeline, ARTIFACTS_DIR / "best_model.joblib")
results_df.to_csv(ARTIFACTS_DIR / "model_metrics.csv", index=False)

config = {
    "model_name": best_model_name,
    "threshold": best_threshold,
    "features": FEATURES,
    "target": TARGET,
}
with open(ARTIFACTS_DIR / "model_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

pred_df = X_test.copy()
pred_df["y_true"] = y_test.values
pred_df["y_proba"] = best_y_proba
pred_df["y_pred"] = best_y_pred
pred_df.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)

# =========================================================
# Enregistrement du meilleur modèle dans MLflow Registry
# + alias "production"
# =========================================================

classifier = best_pipeline.named_steps["classifier"]
preprocessor_fitted = best_pipeline.named_steps["preprocessor"]

if hasattr(preprocessor_fitted, "get_feature_names_out"):
    feature_names = preprocessor_fitted.get_feature_names_out()
else:
    feature_names = None

importance_df = None

if feature_names is not None and hasattr(classifier, "feature_importances_"):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False)

elif feature_names is not None and hasattr(classifier, "coef_"):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(classifier.coef_[0]),
    }).sort_values("importance", ascending=False)

if importance_df is not None:
    importance_df.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)



with mlflow.start_run(run_name="best_model_final"):
    mlflow.log_param("best_model_name", best_model_name)
    mlflow.log_param("threshold", best_threshold)

    mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, best_y_proba))
    mlflow.log_metric("test_pr_auc", average_precision_score(y_test, best_y_proba))
    mlflow.log_metric("test_precision", precision_score(y_test, best_y_pred, zero_division=0))
    mlflow.log_metric("test_recall", recall_score(y_test, best_y_pred, zero_division=0))
    mlflow.log_metric("test_f1", f1_score(y_test, best_y_pred, zero_division=0))

    mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"

mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
wait_until_ready(MODEL_NAME, mv.version)
client.set_registered_model_alias(MODEL_NAME, "production", mv.version)

print(f"\nModèle enregistré dans MLflow Registry : {MODEL_NAME}")
print(f"Alias 'production' -> version {mv.version}")
print("Fichiers sauvegardés dans artifacts/")