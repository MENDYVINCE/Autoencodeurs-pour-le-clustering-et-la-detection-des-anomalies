"""
Application Streamlit - Dashboard de Maintenance Prédictive.
Page d'accueil de l'application.
"""

import streamlit as st
import sys
from pathlib import Path

# --- Configuration du Path ---
src_path = Path(__file__).parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils import local_css

# --- Configuration de la page ---
st.set_page_config(
    page_title="Maintenance Prédictive",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)
local_css("style.css")

# --- Titre Principal ---
st.title("🔧 Dashboard de Maintenance Prédictive")
st.markdown("### Détection d'Anomalies avec Autoencodeurs et Modèles Classiques")
st.divider()

# --- Introduction ---
with st.container():
    st.header("👋 Bienvenue sur le Dashboard")
    st.markdown("""
    Cette application a été conçue pour analyser et prédire les anomalies dans un système industriel
    en utilisant plusieurs approches de Machine Learning. Elle sert d'interface interactive pour explorer les données, 
    tester les modèles en temps réel et visualiser leurs résultats.
    """)

# --- Fonctionnalités Disponibles ---
st.header("🎯 Pages Disponibles")
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.subheader("📊 Visualisation des Données")
        st.markdown("""
        - **Évolution temporelle** des capteurs.
        - **Analyses de corrélation** entre les variables.
        - **Distribution** de l'usure de l'outil.
        """)
with col2:
    with st.container():
        st.subheader("🔮 Prédiction en Temps Réel")
        st.markdown("""
        - **Saisie manuelle** des valeurs de capteurs.
        - **Prédiction instantanée** avec le modèle de votre choix.
        - **Affichage détaillé** du score et de la confiance.
        """)
with col3:
    with st.container():
        st.subheader("🧬 Analyse de l'Espace Latent")
        st.markdown("""
        - **Visualisation PCA/t-SNE** de l'espace appris.
        - **Comparaison** de la séparation des classes (normal vs. anomalie).
        - **Identification** des régimes de fonctionnement.
        """)
st.divider()

# --- Modèles et Performances ---
st.header("🤖 Modèles et Performances")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Modèles Utilisés")
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        st.markdown("""
        **Autoencodeurs (Principaux)**
        - 🧠 Autoencodeur Dense (6 features)
        - 🔄 LSTM Autoencoder (séquences)
        """)
    with sub_col2:
        st.markdown("""
        **Méthodes Classiques (Référence)**
        - 🌲 Isolation Forest
        - 🎯 One-Class SVM
        - 📍 Local Outlier Factor (LOF)
        """)

with col2:
    st.subheader("Meilleur Modèle")
    st.metric(
        "LSTM Autoencoder",
        "F1-Score: 84.8%",
        delta="Recall: 95.9%",
        delta_color="normal"
    )
    st.caption("Le modèle LSTM a montré les meilleures performances pour la détection de pannes.")

st.divider()

# --- Instructions ---
with st.container():
    st.header("🚀 Pour Commencer")
    st.markdown("""
    1.  Naviguez vers la page **📊 Visualisation des Données** pour explorer les capteurs.
    2.  Allez sur **🔮 Prédiction en Temps Réel** pour tester un modèle avec vos propres valeurs.
    3.  Utilisez **🧬 Analyse de l'Espace Latent** pour voir comment les autoencodeurs interprètent les données.
    
    Le jeu de données utilisé est le **AI4I 2020 Predictive Maintenance Dataset**, qui est chargé automatiquement.
    """)

# --- Footer ---
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Projet de Maintenance Prédictive | Technologies : Python • TensorFlow • Scikit-learn • Streamlit</p>
</div>
""", unsafe_allow_html=True)