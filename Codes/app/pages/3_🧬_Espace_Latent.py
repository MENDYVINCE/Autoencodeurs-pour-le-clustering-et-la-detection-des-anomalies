"""
Page Streamlit pour l'analyse de l'espace latent des autoencodeurs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

# --- Configuration du Path ---
# Ajoute le dossier src au path pour importer les modules personnalisés
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils import local_css
from model_loader import load_all_models
from preprocessing import preprocess_for_classic_models, preprocess_for_lstm

# --- Configuration de la page et Style ---
st.set_page_config(
    page_title="Analyse de l'Espace Latent",
    page_icon="🧬",
    layout="wide"
)
local_css("style.css") # Injecter le CSS personnalisé

# --- Chargement des données et modèles (mis en cache) ---
@st.cache_resource
def load_models():
    """Charge et met en cache tous les modèles et scalers."""
    return load_all_models(models_dir='./models')

@st.cache_data
def load_data():
    """Charge et met en cache le dataset."""
    try:
        df = pd.read_csv('data/ai4i2020.csv')
        return df
    except FileNotFoundError:
        st.error("Fichier 'data/ai4i2020.csv' non trouvé.")
        return None

# --- Interface Principale ---
st.title("🧬 Analyse de l'Espace Latent des Autoencodeurs")
st.markdown("""
Cette page vous permet de visualiser comment les modèles d'autoencodeurs "voient" 
les données dans un espace compressé. Les anomalies devraient idéalement former des 
clusters distincts des données normales.
""")

# Charger les données et les modèles
models_data = load_models()
df = load_data()

if df is not None and models_data:
    models = models_data['models']
    scalers = models_data['scalers']
    
    st.sidebar.header("🛠️ Options d'Analyse")
    
    # 2. Sélection du modèle d'autoencodeur
    ae_model_name = st.sidebar.selectbox(
        "1. Choisissez un modèle d'autoencodeur",
        options=['Autoencodeur Dense', 'Autoencodeur LSTM']
    )
    model_key = 'autoencoder' if 'Dense' in ae_model_name else 'lstm'

    # 3. Sélection de la technique de réduction
    reducer_name = st.sidebar.selectbox(
        "2. Choisissez une technique de visualisation",
        options=['PCA', 't-SNE']
    )

    # 4. Bouton de lancement
    analyze_button = st.sidebar.button("🚀 Lancer l'Analyse", use_container_width=True)

    # --- Analyse et Visualisation ---
    if analyze_button:
        try:
            with st.container():
                st.header("Visualisation de l'Espace Latent")
                
                # Garder les labels pour la couleur
                labels = df['Machine failure']

                # Prétraitement
                with st.spinner("Prétraitement des données en cours..."):
                    if model_key == 'autoencoder':
                        data_dicts = df.to_dict(orient='records')
                        processed_list = [preprocess_for_classic_models(d) for d in data_dicts]
                        processed_data = pd.concat(processed_list, ignore_index=True)
                        X_scaled = scalers[model_key].transform(processed_data)
                    else: # LSTM
                        st.warning("Le traitement pour LSTM sur un batch complet peut être approximatif.")
                        data_dicts = df.to_dict(orient='records')
                        processed_sequences = [preprocess_for_lstm(d, timesteps=20) for d in data_dicts]
                        processed_data = np.vstack(processed_sequences)
                        X_scaled = scalers[model_key].transform(processed_data.reshape(-1, processed_data.shape[-1])).reshape(processed_data.shape)

                # Extraction de l'espace latent
                with st.spinner("Extraction de la représentation latente..."):
                    autoencoder_model = models[model_key]
                    try:
                        encoder_output = autoencoder_model.get_layer('espace_latent').output
                        encoder = keras.Model(inputs=autoencoder_model.input, outputs=encoder_output)
                    except ValueError:
                        st.error("Impossible de trouver la couche 'espace_latent'. Utilisation d'une couche intermédiaire.")
                        num_layers = len(autoencoder_model.layers)
                        encoder_output = autoencoder_model.layers[num_layers // 2].output
                        encoder = keras.Model(inputs=autoencoder_model.input, outputs=encoder_output)
                    
                    latent_representation = encoder.predict(X_scaled, verbose=0)
                    if len(latent_representation.shape) == 3:
                        latent_representation = np.mean(latent_representation, axis=1)

                # Réduction de dimension
                with st.spinner(f"Réduction de la dimension avec {reducer_name}..."):
                    if reducer_name == 'PCA':
                        reducer = PCA(n_components=2)
                    else: # t-SNE
                        reducer = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
                    latent_2d = reducer.fit_transform(latent_representation)

                # Création du DataFrame pour le plot
                plot_df = pd.DataFrame(latent_2d, columns=['Composante 1', 'Composante 2'])
                plot_df['Anomalie'] = labels.map({0: 'Normal', 1: 'Anomalie'})

                # Visualisation
                fig = px.scatter(
                    plot_df,
                    x='Composante 1',
                    y='Composante 2',
                    color='Anomalie',
                    title=f"Espace Latent ({ae_model_name}) visualisé avec {reducer_name}",
                    color_discrete_map={'Normal': '#007ACC', 'Anomalie': '#E74C3C'},
                    hover_data={'Anomalie': True}
                )
                fig.update_layout(
                    legend_title_text='Statut de la machine',
                    plot_bgcolor='#2D2D2D',
                    paper_bgcolor='#1E1E1E',
                    font_color='#E0E0E0'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.success("Analyse terminée !")

        except Exception as e:
            st.error(f"Une erreur est survenue durant l'analyse : {e}")
            st.exception(e)
    else:
        st.info("👈 Configurez les options dans le menu latéral et cliquez sur 'Lancer l'Analyse'.")
else:
    st.error("Le chargement des données ou des modèles a échoué. Vérifiez les chemins et les fichiers.")