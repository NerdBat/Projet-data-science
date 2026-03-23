import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, recall_score, f1_score, classification_report
from mlflow.models import infer_signature

# ─────────────────────────────────────────────
# 1. Chargement & nettoyage des données
# ─────────────────────────────────────────────
df = pd.read_csv('../SOURCE/dataset.csv')
df['complaint_type'] = df['complaint_type'].fillna('None')

target = 'churn'
X = df.drop(columns=[target, 'customer_id'])
y = df[target]

# ─────────────────────────────────────────────
# 2. Détection automatique des colonnes
# ─────────────────────────────────────────────
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# ─────────────────────────────────────────────
# 3. Préprocesseur (Scaling + Encoding)
# ─────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ─────────────────────────────────────────────
# 4. Split Train / Test stratifié
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train : {X_train.shape} | Test : {X_test.shape}")

# ─────────────────────────────────────────────
# 5. Définition des modèles (4 dont 1 DL)
# ─────────────────────────────────────────────
models = {
    "Logistic Regression (Baseline)": LogisticRegression(
        max_iter=1000, class_weight='balanced'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    "Deep Learning (MLP)": MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
    ),
}

# ─────────────────────────────────────────────
# 6. Expérience MLflow
# ─────────────────────────────────────────────
mlflow.set_experiment("Churn_Prediction_Project")

best_roc_auc = 0
best_model_name = None
best_model_uri = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        # Pipeline complet (évite le data leakage)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Entraînement
        pipeline.fit(X_train, y_train)

        # Prédictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Métriques
        roc_auc = roc_auc_score(y_test, y_proba)
        recall  = recall_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred)
        report  = classification_report(y_test, y_pred, output_dict=True)
        precision = report['1']['precision']

        # Signature pour l'API
        signature = infer_signature(X_test, y_pred)

        # Logging paramètres & métriques
        mlflow.log_param("model_type", name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("roc_auc",   roc_auc)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("precision", precision)

        # Enregistrement du pipeline (artifact)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:5]
        )

        # Model Registry
        mlflow.register_model(model_info.model_uri, "Churn_Classifier")

        print(f"--- {name} ---")
        print(f"ROC-AUC : {roc_auc:.4f} | Recall : {recall:.4f} | F1 : {f1:.4f}\n")

        # Suivi du meilleur modèle (critère : ROC-AUC)
        if roc_auc > best_roc_auc:
            best_roc_auc    = roc_auc
            best_model_name = name
            best_model_uri  = model_info.model_uri

print("=" * 50)
print(f"✅ Meilleur modèle : {best_model_name} (ROC-AUC = {best_roc_auc:.4f})")
print(f"   URI : {best_model_uri}")
print("=" * 50)
