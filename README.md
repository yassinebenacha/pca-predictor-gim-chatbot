





# 🔧 Prédicteur PCA & Assistant GIM

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Description

Système d'intelligence artificielle intégré combinant :

1. **🎯 Prédicteur PCA** : Prédiction automatique des solutions techniques (PCA) pour diagnostic automobile
2. **🤖 Assistant GIM** : Chatbot intelligent pour la plateforme Global Issue Management

## 🚀 Fonctionnalités Principales

### 🔧 **Prédicteur PCA**
- ✅ **Prédiction IA** de solutions techniques à partir de codes DTC
- ✅ **Dataset augmenté** : 17,800 exemples, 178 classes PCA
- ✅ **Accuracy 23%** sur classification multi-classe complexe
- ✅ **Interface Streamlit** intuitive pour techniciens

### 🤖 **Assistant GIM**
- ✅ **Chatbot intelligent** intégré à l'interface
- ✅ **API Generative Engine** Capgemini (GPT-4)
- ✅ **Mode démo** fonctionnel sans configuration
- ✅ **Expertise complète** sur la plateforme GIM

## 🏗️ Architecture

```
├── 📊 Machine Learning
│   ├── preprocessing.py      # Préprocessing NLP
│   ├── train_model.py       # TF-IDF + RandomForest
│   └── predict.py           # Prédictions PCA
├── 🤖 Chatbot GIM
│   ├── gim_chatbot.py       # Assistant intelligent
│   └── config.py            # Configuration API
├── 🌐 Interface
│   └── app.py               # Application Streamlit
└── 📁 Données
    ├── data/                # Datasets
    └── models/              # Modèles entraînés
```

## 🚀 Installation Rapide

### 1. **Cloner le Repository**
```bash
git clone https://github.com/yassinebenacha/pca-predictor-gim-chatbot.git
cd pca-predictor-gim-chatbot
```

### 2. **Installer les Dépendances**
```bash
pip install -r requirements.txt
```

### 3. **Configuration (Optionnel)**
```bash
# Copier le template de configuration
cp .env.example .env

# Éditer .env avec votre clé API
GENERATIVE_ENGINE_API_KEY=votre-clé-api
```

### 4. **Entraîner le Modèle**
```bash
python main.py --action train
```

### 5. **Lancer l'Application**
```bash
streamlit run app.py
```

## 💡 Utilisation

### 🎯 **Prédiction PCA**
1. Saisissez le **Code DTC** (ex: P0420)
2. Décrivez le **problème observé**
3. Ajoutez la **cause racine** (optionnel)
4. Obtenez la **PCA recommandée** avec niveau de confiance

### 🤖 **Assistant GIM**
1. Cliquez sur **"🤖 Ouvrir l'Assistant GIM"**
2. Posez vos questions sur la plateforme GIM
3. Consultez l'historique des conversations

## 📊 Performances

- **🎯 Accuracy PCA** : 23% sur 178 classes
- **📈 Dataset** : 17,800 exemples équilibrés
- **🤖 Chatbot** : 10 questions suggérées + mode démo
- **⚡ Temps de réponse** : < 2 secondes

## 🔧 Technologies

- **Python 3.8+** : Langage principal
- **Streamlit** : Interface web interactive
- **Scikit-learn** : Machine Learning (TF-IDF + RandomForest)
- **OpenAI API** : Chatbot intelligent
- **Pandas/NumPy** : Manipulation de données
- **Plotly** : Visualisations interactives

## 📁 Structure du Projet

```
pca-predictor-gim-chatbot/
├── 📊 Modules ML
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── main.py
├── 🤖 Chatbot GIM
│   ├── gim_chatbot.py
│   └── config.py
├── 🌐 Interface
│   └── app.py
├── 📁 Données
│   ├── data/
│   └── models/
├── 🧪 Tests & Démo
│   ├── demo.py
│   ├── demo_complet.py
│   ├── test_gim_integration.py
│   └── validation_finale.py
├── 📖 Documentation
│   ├── README.md
│   ├── GUIDE_CHATBOT_GIM.md
│   └── INTEGRATION_COMPLETE.md
└── ⚙️ Configuration
    ├── requirements.txt
    ├── .env.example
    └── .gitignore
```

## 🎯 Cas d'Usage

### 👨‍🔧 **Technicien Automobile**
- Diagnostic rapide avec codes DTC
- Recherche de solutions PCA
- Formation sur les processus GIM

### 👨‍💼 **Ingénieur Qualité**
- Analyse des tendances de pannes
- Optimisation des processus
- Formation équipe sur GIM

### 👨‍💻 **Manager Opérationnel**
- Supervision des diagnostics
- Métriques de performance
- Gestion des incidents GIM

## 🤝 Contribution

1. **Fork** le projet
2. **Créez** une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. **Committez** vos changements (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. **Poussez** vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. **Ouvrez** une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🆘 Support

- 📖 **Documentation** : Consultez les guides dans `/docs`
- 🐛 **Issues** : Ouvrez une issue sur GitHub
- 💬 **Questions** : Utilisez les discussions GitHub

## 🎉 Remerciements

- **Capgemini** pour l'API Generative Engine
- **Équipe Diagnostic Automobile** pour les spécifications
- **Communauté Open Source** pour les outils utilisés

---

**🚗 Prêt à révolutionner votre diagnostic automobile avec l'IA !** 🔧🤖
