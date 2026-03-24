from pathlib import Path
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# Configuration
# =========================================================
st.set_page_config(
    page_title="Dashboard Churn",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

DEFAULT_API_URL = "http://127.0.0.1:8000"


# =========================================================
# Fonctions utiles
# =========================================================
@st.cache_data
def load_metrics():
    path = ARTIFACTS_DIR / "model_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_feature_importance():
    path = ARTIFACTS_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def get_api_health(api_url):
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


def get_model_info(api_url):
    try:
        response = requests.get(f"{api_url}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


def predict_churn(api_url, payload):
    response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


# =========================================================
# Titre
# =========================================================
st.title("Dashboard de prédiction du churn client")
st.markdown("Simulation d’un client, comparaison des modèles et visualisation des variables importantes.")


# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Configuration")

api_url = st.sidebar.text_input("URL de l'API", value=DEFAULT_API_URL)

health = get_api_health(api_url)
if health:
    st.sidebar.success(f"API connectée ({health.get('model_source', 'unknown')})")
else:
    st.sidebar.error("API non disponible")

model_info = get_model_info(api_url)
if model_info:
    st.sidebar.markdown("### Modèle utilisé")
    st.sidebar.write(f"**Nom :** {model_info.get('model_name', 'N/A')}")
    st.sidebar.write(f"**Seuil :** {model_info.get('threshold', 'N/A')}")


# =========================================================
# Onglets
# =========================================================
tab1, tab2, tab3 = st.tabs(["Prédiction client", "Comparaison des modèles", "Variables importantes"])


# =========================================================
# Onglet 1 : Prédiction
# =========================================================
with tab1:
    st.subheader("Simulation d’un client")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Genre", ["Male", "Female"])
        age = st.number_input("Âge", min_value=18, max_value=100, value=35)
        tenure_months = st.number_input("Ancienneté (mois)", min_value=0, value=12)
        contract_type = st.selectbox("Type de contrat", ["Monthly", "Quarterly", "Yearly"])
        monthly_fee = st.number_input("Frais mensuels (€)", min_value=0.0, value=49.99, step=1.0)

    with col2:
        monthly_logins = st.number_input("Connexions mensuelles", min_value=0, value=20)
        weekly_active_days = st.number_input("Jours actifs par semaine", min_value=0, max_value=7, value=3)
        avg_session_time = st.number_input("Temps moyen de session (minutes)", min_value=0.0, value=25.0, step=1.0)
        payment_failures = st.number_input("Échecs de paiement", min_value=0, value=0)
        nps_score = st.slider("NPS", min_value=-100, max_value=100, value=0)

    payload = {
        "gender": gender,
        "age": int(age),
        "tenure_months": int(tenure_months),
        "contract_type": contract_type,
        "monthly_logins": int(monthly_logins),
        "weekly_active_days": int(weekly_active_days),
        "avg_session_time": float(avg_session_time),
        "monthly_fee": float(monthly_fee),
        "payment_failures": int(payment_failures),
        "nps_score": int(nps_score),
    }

    if st.button("Lancer la prédiction", use_container_width=True):
        try:
            result = predict_churn(api_url, payload)

            proba = result["churn_probability"]
            prediction = result["prediction"]
            threshold = result["threshold_used"]
            risk_level = result["risk_level"]

            st.success("Prédiction réalisée avec succès")

            c1, c2, c3 = st.columns(3)
            c1.metric("Probabilité de churn", f"{proba:.1%}")
            c2.metric("Seuil utilisé", f"{threshold:.2f}")
            c3.metric("Niveau de risque", risk_level.capitalize())

            if prediction == 1:
                st.error("Le client est prédit comme susceptible de churner.")
            else:
                st.success("Le client est prédit comme non churner.")

            # estimation simple du revenu à risque
            annual_revenue = monthly_fee * 12
            revenue_at_risk = annual_revenue * proba

            c4, c5 = st.columns(2)
            c4.metric("Revenu annuel estimé", f"{annual_revenue:.2f} €")
            c5.metric("Revenu estimé à risque", f"{revenue_at_risk:.2f} €")

            # jauge de risque
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Risque de churn (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 30], "color": "#d4edda"},
                            {"range": [30, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#f8d7da"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # explication métier simple
            st.markdown("### Lecture métier rapide")
            reasons = []

            if contract_type == "Monthly":
                reasons.append("- Contrat mensuel : plus de flexibilité, donc plus de risque de départ.")
            if nps_score <= 4:
                reasons.append("- NPS faible : signe d’insatisfaction potentielle.")
            if payment_failures >= 1:
                reasons.append("- Présence d’échecs de paiement.")
            if weekly_active_days <= 2:
                reasons.append("- Faible activité hebdomadaire.")
            if monthly_logins < 10:
                reasons.append("- Peu de connexions mensuelles.")
            if tenure_months < 6:
                reasons.append("- Client récent, potentiellement moins fidélisé.")

            if reasons:
                st.write("Facteurs de risque possibles :")
                for r in reasons:
                    st.write(r)
            else:
                st.write("Le profil ne présente pas de facteur de risque évident parmi les règles métier simples du dashboard.")

        except requests.exceptions.ConnectionError:
            st.error("Impossible de contacter l’API. Vérifie que FastAPI est bien lancée.")
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = e.response.json()
            except Exception:
                error_detail = str(e)
            st.error(f"Erreur API : {error_detail}")
        except Exception as e:
            st.error(f"Erreur : {e}")


# =========================================================
# Onglet 2 : Comparaison des modèles
# =========================================================
with tab2:
    st.subheader("Comparaison des modèles")

    metrics_df = load_metrics()

    if metrics_df is None:
        st.warning("Le fichier artifacts/model_metrics.csv est introuvable. Lance train.py d’abord.")
    else:
        st.dataframe(metrics_df, use_container_width=True)

        metric_choice = st.selectbox(
            "Choisir une métrique à afficher",
            [
                "cv_roc_auc_mean",
                "cv_pr_auc_mean",
                "cv_recall_mean",
                "cv_f1_mean",
                "test_roc_auc",
                "test_pr_auc",
                "test_recall",
                "test_f1",
            ]
        )

        fig_models = px.bar(
            metrics_df.sort_values(metric_choice, ascending=False),
            x="model",
            y=metric_choice,
            text=metric_choice,
            title=f"Comparaison des modèles selon {metric_choice}",
        )
        fig_models.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig_models, use_container_width=True)

        st.markdown("### Interprétation")
        best_row = metrics_df.sort_values(metric_choice, ascending=False).iloc[0]
        st.write(
            f"Selon la métrique **{metric_choice}**, le meilleur modèle est "
            f"**{best_row['model']}** avec un score de **{best_row[metric_choice]:.4f}**."
        )


# =========================================================
# Onglet 3 : Importance des variables
# =========================================================
with tab3:
    st.subheader("Variables importantes")

    importance_df = load_feature_importance()

    if importance_df is None:
        st.warning(
            "Le fichier artifacts/feature_importance.csv est introuvable. "
            "Il est créé seulement si le modèle final fournit des importances."
        )
    else:
        st.dataframe(importance_df.head(15), use_container_width=True)

        top_n = st.slider("Nombre de variables à afficher", min_value=5, max_value=20, value=10)

        top_features = importance_df.head(top_n).sort_values("importance", ascending=True)

        fig_importance = px.bar(
            top_features,
            x="importance",
            y="feature",
            orientation="h",
            title=f"Top {top_n} variables les plus importantes",
        )
        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("### Utilité métier")
        st.write(
            "Cette vue permet d’identifier les variables qui influencent le plus la prédiction "
            "du modèle final. Cela peut aider à cibler les actions de rétention client."
        )