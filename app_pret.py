import streamlit as st
import pandas as pd
import joblib
import os

# Chemin vers le modèle
model_path = r"xgboost_loan_model.joblib"

# Chargement du modèle avec gestion d'erreur
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"❌ Fichier modèle non trouvé : {model_path}")
    st.stop()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")
    st.stop()

# Configuration de la page
st.set_page_config(page_title="Prédiction de prêt bancaire", page_icon="💰")

st.title("📊 Prédiction d'approbation de prêt bancaire")
st.markdown("Remplissez les informations du client ci-dessous pour obtenir une prédiction.")

# Saisie utilisateur (catégories en clair, sans encodage manuel)
person_age = st.slider("Âge du client", 18, 80, 30)
person_income = st.number_input("Revenu mensuel (€)", value=3000)
person_home_ownership = st.selectbox("Statut du logement", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.slider("Ancienneté de l’emploi (années)", 0, 40, 5)
person_emp_exp = st.slider("Expérience professionnelle (années)", 0, 40, 5)
person_education = st.selectbox("Niveau d'études", ["PRIMARY", "SECONDARY", "BACHELORS", "MASTERS", "DOCTORATE"])
person_gender = st.selectbox("Sexe", ["Male", "Female", "Other"])
loan_amnt = st.number_input("Montant du prêt (€)", value=10000)
loan_int_rate = st.slider("Taux d’intérêt (%)", 0.0, 40.0, 12.0)
loan_percent_income = st.slider("Part du revenu utilisé pour rembourser (%)", 0.0, 100.0, 30.0)
loan_intent = st.selectbox("But du prêt", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
cb_person_default_on_file = st.selectbox("Défaut de paiement dans le passé ?", ["Yes", "No"])
previous_loan_defaults_on_file = st.selectbox("Prêts précédents en défaut ?", ["Yes", "No"])
credit_score = st.number_input("Score de crédit", value=600)
cb_person_cred_hist_length = st.slider("Ancienneté de l’historique de crédit (années)", 0, 50, 10)

# Création du DataFrame d'entrée (avec les mêmes colonnes que lors de l'entraînement)
input_df = pd.DataFrame({
    "person_age": [person_age],
    "person_income": [person_income],
    "person_home_ownership": [person_home_ownership],
    "person_emp_length": [person_emp_length],
    "person_emp_exp": [person_emp_exp],
    "person_education": [person_education],
    "person_gender": [person_gender],
    "loan_amnt": [loan_amnt],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "loan_intent": [loan_intent],
    "cb_person_default_on_file": [cb_person_default_on_file],
    "previous_loan_defaults_on_file": [previous_loan_defaults_on_file],
    "credit_score": [credit_score],
    "cb_person_cred_hist_length": [cb_person_cred_hist_length]
})

# Bouton de prédiction
if st.button("🔮 Prédire le statut du prêt"):
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
        st.stop()

    if prediction == 1:
        st.success(f"✅ Le prêt est **approuvé** avec une probabilité de {prediction_proba:.2%}")
        st.balloons()
        st.markdown("### 🥳 Félicitations ! Votre demande de prêt a été acceptée.")
    else:
        st.error(f"❌ Le prêt est **refusé** avec une probabilité de {(1 - prediction_proba):.2%}")
        st.snow()
        st.markdown("### 💡 Conseil : Améliorez votre dossier et retentez votre chance.")
