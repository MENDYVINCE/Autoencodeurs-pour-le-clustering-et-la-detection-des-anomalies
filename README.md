## Application en Maintenance Prédictive

---

## 📑 Table des Matières

- [📌 Description du projet](#-description-du-projet)
- [🎯 Objectifs](#-objectifs)
- [🧠 Approches utilisées](#-approches-utilisées)
- [🧪 Tâches principales](#-tâches-principales)
- [📊 Tableau de bord Streamlit](#-tableau-de-bord-streamlit)
- [🛠️ Technologies utilisées](#️-technologies-utilisées)
- [📁 Jeu de données](#-jeu-de-données)
- [📦 Installation](#-installation)
- [🚀 Exécution Locale](#-exécution-locale)
- [🌐 Application Déployée](#-application-déployée)
- [📦 Livrables](#-livrables)

---

## 📌 Description du projet

Ce projet vise à **concevoir, implémenter et comparer plusieurs approches de détection d’anomalies** dans un contexte industriel de **maintenance prédictive**, en s’appuyant principalement sur les **autoencodeurs** (réseaux de neurones non supervisés).

L’objectif est d’apprendre les **comportements normaux des machines** à partir de données de capteurs, puis de détecter les **déviations anormales** annonçant des défaillances.

---

## 🎯 Objectifs

* Détecter automatiquement les anomalies dans des données industrielles
* Comparer les **autoencodeurs** avec des méthodes classiques
* Identifier des **régimes de fonctionnement** via le clustering
* Proposer une **solution automatisée et déployée**

---

## 🧠 Approches utilisées

### 1️⃣ Méthodes principales – Autoencodeurs

* **Autoencodeur Dense (Feedforward AE)**

  * Apprentissage d’une représentation compacte
  * Détection par **erreur de reconstruction**
* **Autoencodeur LSTM**

  * Adapté aux **séries temporelles**
  * Détection d’anomalies séquentielles (pré-pannes)

### 2️⃣ Méthodes classiques (comparaison)

* **Isolation Forest**
* **One-Class SVM**
* **Local Outlier Factor (LOF)**

---

## 🧪 Tâches principales

1. Fondements théoriques
2. Prétraitement et analyse exploratoire (EDA)
3. Modélisation avec autoencodeurs
4. Comparaison avec méthodes classiques
5. **Clustering dans l’espace latent**

   * K-Means
   * DBSCAN
   * Régimes : *Normal / Dégradé / En panne*
6. Visualisation et interprétation

   * Erreur de reconstruction
   * Anomalies par méthode
   * Analyse du latent space (PCA, t-SNE)
7. Développement d’un **dashboard Streamlit**

---

## 📊 Tableau de bord Streamlit

Le tableau de bord affiche :

* Courbes des capteurs
* Anomalies détectées par chaque méthode
* Clusters issus de l’espace latent
* Comparaisons visuelles interactives

---

## 🛠️ Technologies utilisées

* **Python**
* **TensorFlow / Keras**
* **Scikit-learn**
* **Pandas, NumPy**
* **Matplotlib, Seaborn, Plotly**
* **Streamlit**

---

## 📁 Jeu de données

**AI4I 2020 Predictive Maintenance Dataset**
🔗 [https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

---

## 📦 Livrables

1. 📄 Rapport technique
2. 💻 Code source du projet
3. 🎥 Présentation vidéo avec démonstration (15–20 min)
4. 📊 Présentation PowerPoint

---

## � Installation

### Prérequis

- Python 3.10 ou supérieur
- Poetry (gestionnaire de dépendances)

### Installation des Dépendances

Pour exécuter le **notebook Jupyter** :

```bash
pip install -r requirements.txt
```

Les dépendances principales incluent :
- `pandas>=2.3.3`
- `numpy>=2.4.0`
- `scikit-learn>=1.8.0`
- `tensorflow>=2.20.0`
- `keras>=3.13.0`
- `matplotlib>=3.10.8`
- `seaborn>=0.13.2`
- `plotly>=6.5.0`
- `streamlit>=1.52.2`
- `joblib>=1.5.3`
- `pyyaml>=6.0.3`

---

## 🚀 Exécution Locale

### Lancer l'Application Streamlit

1. **Activez l'environnement Poetry** :
```bash
cd Codes/app
poetry shell
```

2. **Lancez l'application** :
```bash
streamlit run app.py
```

3. **Accédez à l'application** :
Ouvrez votre navigateur à l'adresse : `http://localhost:8501`

---

## 🌐 Application Déployée

L'application est déployée sur **Streamlit Cloud** et accessible en ligne :

🔗 **[https://maintenancepredictive.streamlit.app/](https://maintenancepredictive.streamlit.app/)**

### Fonctionnalités Disponibles

- 📊 **Visualisation des Capteurs** : Évolution temporelle et corrélations
- 🔮 **Prédiction en Temps Réel** : Testez les modèles avec vos propres données
- 🧬 **Analyse de l'Espace Latent** : Clustering et visualisation PCA/t-SNE
- 🏭 **Simulation de Production** : Simulation en temps réel avec détection d'anomalies

---

## 📦 Livrables

1. 📄 Rapport technique
2. 💻 Code source du projet
3. 🎥 Présentation vidéo avec démonstration (15–20 min)
4. 📊 Présentation PowerPoint

---
