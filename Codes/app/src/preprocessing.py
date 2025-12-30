"""
Module de prétraitement et feature engineering pour la maintenance prédictive.

Ce module contient les fonctions pour transformer les données brutes des capteurs
en features utilisables par les modèles de détection d'anomalies.
"""

import pandas as pd
import numpy as np


def create_features(df):
    """
    Applique le feature engineering optimisé (uniquement les 6 features utilisées).
    
    Paramètres :
    -----------
    df : pd.DataFrame
        DataFrame avec les colonnes de base :
        - Air temperature [K]
        - Process temperature [K]
        - Rotational speed [rpm]
        - Torque [Nm]
        - Tool wear [min]
    
    Retourne :
    ---------
    pd.DataFrame
        DataFrame avec les 6 features optimales :
        - Tool wear [min] (original)
        - temp_diff
        - power_kw
        - high_wear
        - Torque [Nm] (original)
        - mechanical_stress
    """
    # Copie pour ne pas modifier l'original
    df_processed = df.copy()
    
    # ============================================
    # FEATURES NÉCESSAIRES (6 au total)
    # ============================================
    
    # 1. Tool wear [min] - Déjà présent (original)
    
    # 2. temp_diff - Différence de température
    df_processed['temp_diff'] = (
        df_processed['Process temperature [K]'] - df_processed['Air temperature [K]']
    )
    
    # 3. power_kw - Puissance en kW
    df_processed['power_kw'] = (
        df_processed['Torque [Nm]'] * df_processed['Rotational speed [rpm]']
    ) / 9550
    
    # 4. high_wear - Usure élevée (binaire)
    df_processed['high_wear'] = (df_processed['Tool wear [min]'] > 200).astype(int)
    
    # 5. Torque [Nm] - Déjà présent (original)
    
    # 6. mechanical_stress - Stress mécanique
    df_processed['mechanical_stress'] = (
        df_processed['Torque [Nm]'] * (1 + df_processed['Tool wear [min]'] / 250)
    )
    
    return df_processed


def select_features(df, feature_list=None):
    """
    Sélectionne uniquement les features utilisées par les modèles.
    
    Paramètres :
    -----------
    df : pd.DataFrame
        DataFrame avec toutes les features
    feature_list : list, optional
        Liste des features à sélectionner. Si None, utilise les 6 features optimales.
    
    Retourne :
    ---------
    pd.DataFrame
        DataFrame avec uniquement les features sélectionnées
    """
    if feature_list is None:
        # Features optimales identifiées lors de l'analyse
        feature_list = [
            'Tool wear [min]',
            'temp_diff',
            'power_kw',
            'high_wear',
            'Torque [Nm]',
            'mechanical_stress'
        ]
    
    # Vérifier que toutes les features existent
    missing_features = [f for f in feature_list if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features manquantes : {missing_features}")
    
    return df[feature_list]


def preprocess_for_prediction(data_dict):
    """
    Prétraite des données brutes pour la prédiction.
    
    Paramètres :
    -----------
    data_dict : dict
        Dictionnaire avec les valeurs des capteurs :
        {
            'Air temperature [K]': float,
            'Process temperature [K]': float,
            'Rotational speed [rpm]': float,
            'Torque [Nm]': float,
            'Tool wear [min]': float,
            'Type': str (optionnel)
        }
    
    Retourne :
    ---------
    pd.DataFrame
        DataFrame prêt pour la prédiction (1 ligne, features sélectionnées)
    """
    # Créer DataFrame à partir du dictionnaire
    df = pd.DataFrame([data_dict])
    
    # Feature engineering
    df_features = create_features(df)
    
    # Sélectionner les features optimales
    df_selected = select_features(df_features)
    
    return df_selected



# ============================================
# EXEMPLE D'UTILISATION
# ============================================

if __name__ == "__main__":
    # Test avec des données d'exemple
    test_data = {
        'Air temperature [K]': 298.1,
        'Process temperature [K]': 308.6,
        'Rotational speed [rpm]': 1551,
        'Torque [Nm]': 42.8,
        'Tool wear [min]': 0,
        'Type': 'M'
    }
    
    print("Test du preprocessing...")
    result = preprocess_for_prediction(test_data)
    print(f"\nFeatures générées :")
    print(result)
    print(f"\nShape : {result.shape}")
    print(f"Colonnes : {list(result.columns)}")
