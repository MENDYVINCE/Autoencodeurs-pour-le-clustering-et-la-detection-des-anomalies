# 🔧 Autoencodeurs pour le Clustering et la Détection d’Anomalies

## Application en Maintenance Prédictive

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

## 🚀 Déploiement

* Développement et entraînement : **Google Colab / environnement local**
* Déploiement : **Streamlit (local ou cloud)**

---

Si tu veux, je peux aussi te fournir :

* 📂 **une structure de dossiers**
* 🧾 **un README plus court**
* 📊 **un plan de rapport technique**
* 🎤 **un script pour la présentation vidéo**
