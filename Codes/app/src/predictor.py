"""
Module de prédiction avec les modèles de détection d'anomalies.

Ce module contient les fonctions pour faire des prédictions avec chaque modèle.
"""

import numpy as np


def predict_with_autoencoder(X, model, scaler, threshold=None):
    """
    Prédit avec l'autoencodeur (basé sur MSE).
    
    Paramètres :
    -----------
    X : array-like
        Features (non normalisées)
    model : keras.Model
        Autoencodeur chargé
    scaler : StandardScaler
        Scaler pour normaliser X
    threshold : float, optional
        Seuil MSE pour considérer comme anomalie.
        Si None, utilise le seuil optimisé (80ème percentile des normaux).
        Valeur optimale trouvée lors de l'entraînement : ~0.45-0.50
    
    Retourne :
    ---------
    dict
        {
            'prediction': int (0=Normal, 1=Panne),
            'mse': float,
            'confidence': float
        }
    """
    # Seuil optimisé (80ème percentile des données normales d'entraînement)
    # Résultats : Recall=0.894, Precision=0.136, F1=0.235
    if threshold is None:
        threshold = 0.48  # Valeur approximative du 80ème percentile
    
    # Normaliser
    X_scaled = scaler.transform(X)
    
    # Reconstruction
    X_reconstructed = model.predict(X_scaled, verbose=0)
    
    # Calculer MSE
    mse = np.mean(np.square(X_scaled - X_reconstructed), axis=1)[0]
    
    # Prédiction
    prediction = 1 if mse > threshold else 0
    
    # Confiance (distance au seuil)
    confidence = min(abs(mse - threshold) / threshold * 100, 100)
    
    return {
        'prediction': prediction,
        'mse': float(mse),
        'confidence': float(confidence),
        'threshold': threshold
    }


def predict_with_isolation_forest(X, model, scaler):
    """
    Prédit avec Isolation Forest.
    
    Paramètres :
    -----------
    X : array-like
        Features (non normalisées)
    model : IsolationForest
        Modèle chargé
    scaler : StandardScaler
        Scaler pour normaliser X
    
    Retourne :
    ---------
    dict
        {
            'prediction': int (0=Normal, 1=Panne),
            'score': float,
            'confidence': float
        }
    """
    # Normaliser
    X_scaled = scaler.transform(X)
    
    # Prédiction (1=Normal, -1=Anomalie)
    pred = model.predict(X_scaled)[0]
    prediction = 0 if pred == 1 else 1
    
    # Score d'anomalie
    score = -model.decision_function(X_scaled)[0]
    
    # Confiance
    confidence = min(abs(score) * 100, 100)
    
    return {
        'prediction': prediction,
        'score': float(score),
        'confidence': float(confidence)
    }


def predict_with_ocsvm(X, model, scaler):
    """
    Prédit avec One-Class SVM.
    
    Paramètres :
    -----------
    X : array-like
        Features (non normalisées)
    model : OneClassSVM
        Modèle chargé
    scaler : StandardScaler
        Scaler pour normaliser X
    
    Retourne :
    ---------
    dict
        {
            'prediction': int (0=Normal, 1=Panne),
            'score': float,
            'confidence': float
        }
    """
    # Normaliser
    X_scaled = scaler.transform(X)
    
    # Prédiction (1=Normal, -1=Anomalie)
    pred = model.predict(X_scaled)[0]
    prediction = 0 if pred == 1 else 1
    
    # Score
    score = model.decision_function(X_scaled)[0]
    
    # Confiance
    confidence = min(abs(score) * 100, 100)
    
    return {
        'prediction': prediction,
        'score': float(score),
        'confidence': float(confidence)
    }


def predict_with_lof(X, model, scaler):
    """
    Prédit avec LOF.
    
    Paramètres :
    -----------
    X : array-like
        Features (non normalisées)
    model : LocalOutlierFactor
        Modèle chargé
    scaler : StandardScaler
        Scaler pour normaliser X
    
    Retourne :
    ---------
    dict
        {
            'prediction': int (0=Normal, 1=Panne),
            'score': float,
            'confidence': float
        }
    """
    # Normaliser
    X_scaled = scaler.transform(X)
    
    # LOF nécessite fit_predict
    pred = model.fit_predict(X_scaled)[0]
    prediction = 0 if pred == 1 else 1
    
    # Score LOF
    score = -model.negative_outlier_factor_[0]
    
    # Confiance
    confidence = min(abs(score - 1) * 100, 100)
    
    return {
        'prediction': prediction,
        'score': float(score),
        'confidence': float(confidence)
    }


def predict_all_models(X, models_dict, models_to_use=None):
    """
    Fait des prédictions avec tous les modèles sélectionnés.
    
    Paramètres :
    -----------
    X : pd.DataFrame
        Features (non normalisées)
    models_dict : dict
        Dictionnaire retourné par load_all_models()
    models_to_use : list, optional
        Liste des modèles à utiliser. Si None, utilise tous.
    
    Retourne :
    ---------
    dict
        {
            'autoencoder': {...},
            'isolation_forest': {...},
            'ocsvm': {...},
            'lof': {...}
        }
    """
    models = models_dict['models']
    scalers = models_dict['scalers']
    
    results = {}
    
    # Autoencodeur
    if (models_to_use is None or 'Autoencodeur' in models_to_use) and 'autoencoder' in models:
        results['autoencoder'] = predict_with_autoencoder(
            X, models['autoencoder'], scalers['autoencoder']
        )
    
    # Isolation Forest
    if (models_to_use is None or 'Isolation Forest' in models_to_use) and 'isolation_forest' in models:
        results['isolation_forest'] = predict_with_isolation_forest(
            X, models['isolation_forest'], scalers['isolation_forest']
        )
    
    # One-Class SVM
    if (models_to_use is None or 'One-Class SVM' in models_to_use) and 'ocsvm' in models:
        results['ocsvm'] = predict_with_ocsvm(
            X, models['ocsvm'], scalers['ocsvm']
        )
    
    # LOF
    if (models_to_use is None or 'LOF' in models_to_use) and 'lof' in models:
        results['lof'] = predict_with_lof(
            X, models['lof'], scalers['lof']
        )
    
    return results
