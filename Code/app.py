import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Churn Decision Support", layout="wide")

# 1. Titre et Description
st.title("🚀 Plateforme Décisionnelle - Rétention Client")
st.markdown("""
Cet outil permet aux responsables marketing et CRM d'anticiper le risque de résiliation 
et d'évaluer l'impact financier en temps réel.
""")

# 2. Barre latérale : Saisie des données
st.sidebar.header("🛠️ Simulation de scénario client")

def user_input_features():
    # Variables clés identifiées dans le dataset
    gender = st.sidebar.selectbox("Genre", ["Male", "Female"])
    age = st.sidebar.slider("Âge", 18, 90, 35)
    tenure = st.sidebar.slider("Ancienneté (mois)", 1, 72, 12)
    contract = st.sidebar.selectbox("Type de contrat", ['Month-to-month', 'One year', 'Two year'])
    monthly_logins = st.sidebar.slider("Connexions mensuelles", 0, 30, 15)
    weekly_days = st.sidebar.slider("Jours actifs / semaine", 0, 7, 3)
    avg_session = st.sidebar.number_input("Durée session moy. (min)", 5.0, 120.0, 30.0)
    monthly_fee = st.sidebar.number_input("Frais Mensuels ($)", 20.0, 200.0, 50.0)
    payment_fail = st.sidebar.slider("Échecs de paiement", 0, 5, 0)
    nps_score = st.sidebar.slider("Score NPS (Satisfaction)", 0, 10, 7)
    
    data = {
        'gender': gender, 
        'age': age, 
        'tenure_months': tenure,
        'contract_type': contract, 
        'monthly_logins': monthly_logins,
        'weekly_active_days': weekly_days, 
        'avg_session_time': avg_session,
        'monthly_fee': monthly_fee, 
        'payment_failures': payment_fail,
        'nps_score': nps_score
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 3. Zone d'affichage principale
col_data, col_pred = st.columns([1, 1])

with col_data:
    st.subheader("📊 Profil du Client")
    st.write(input_df)

with col_pred:
    st.subheader("🔮 Prédiction en temps réel")
    
    if st.button("Lancer l'analyse du risque"):
        try:
            # Appel à l'API FastAPI
            api_url = "http://localhost:8000/predict"
            # On convertit le dataframe en JSON pour l'API
            payload = input_df.to_dict(orient='records')[0]
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                proba = result['churn_probability']
                
                # Calcul du revenu à risque (Business KPI)
                rev_risk = input_df['monthly_fee'].iloc[0] * proba
                
                # Affichage des métriques
                m1, m2 = st.columns(2)
                m1.metric("Probabilité de Churn", f"{proba*100:.1f}%")
                m2.metric("Revenu à risque mensuel", f"{rev_risk:.2f} $")
                
                # Alerte visuelle basée sur le seuil
                if proba > 0.3:
                    st.error("⚠️ Alerte Churn : Ce client présente un risque élevé.")
                    st.info("💡 Action suggérée : Proposer une remise ou un appel de fidélisation.")
                else:
                    st.success("✅ Client Fidèle : Risque de départ négligeable.")
            else:
                st.warning(f"Erreur API ({response.status_code}) : {response.text}")
        
        except Exception as e:
            st.error(f"Erreur de connexion : {e}")

# 4. Section Analyse Globale
st.divider()
st.subheader("📈 Contexte Business")
chart_data = pd.DataFrame({'Statut': ['Fidèles', 'À Risque'], 'Volume': [80, 20]})
fig = px.pie(chart_data, values='Volume', names='Statut', title="Distribution estimée du Churn")
st.plotly_chart(fig)