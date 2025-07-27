# 🤖 Guide d'Utilisation - Chatbot GIM Intégré

## 🎯 Vue d'Ensemble

L'application **Prédicteur PCA** a été enrichie avec un **Assistant GIM intelligent** qui vous aide à comprendre et utiliser la plateforme Global Issue Management directement depuis l'interface Streamlit.

---

## 🚀 Fonctionnalités du Chatbot GIM

### 📋 **Domaines d'Expertise**
- ✅ **Statuts et couleurs GIM** (Rouge, Jaune, Vert, Bleu, etc.)
- ✅ **Processus de traitement** (ICA, RCA, PCA, Clean Point)
- ✅ **Rôles et permissions** (Issue Manager, Business Supervisor, etc.)
- ✅ **Flux de travail** (création, investigation, résolution, clôture)
- ✅ **Intégrations système** (Palantir, eSupplier, CQI, etc.)

### 🎨 **Modes de Fonctionnement**

#### 🌐 **Mode API (Production)**
- Utilise l'API Generative Engine de Capgemini
- Réponses intelligentes et contextuelles
- Apprentissage continu des conversations

#### 🎭 **Mode Démo (Développement)**
- Réponses prédéfinies pour les questions courantes
- Fonctionne sans configuration API
- Parfait pour les tests et démonstrations

---

## 🔧 Configuration

### 1. **Installation des Dépendances**
```bash
pip install openai python-dotenv
```

### 2. **Configuration de l'API (Optionnel)**
```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer le fichier .env
GENERATIVE_ENGINE_API_KEY=votre-clé-api-ici
```

### 3. **Lancement de l'Application**
```bash
streamlit run app.py
```

---

## 💡 Comment Utiliser le Chatbot

### 🎯 **Accès au Chatbot**

#### **Option 1 : Depuis la Sidebar**
1. Ouvrez l'application Streamlit
2. Dans la sidebar, cliquez sur **"💬 Ouvrir le Chat GIM"**
3. Ou utilisez les **questions rapides** directement

#### **Option 2 : Depuis l'Onglet Principal**
1. Cliquez sur le bouton **"🤖 Ouvrir l'Assistant GIM"** 
2. Un nouvel onglet **"Assistant GIM"** apparaît
3. Naviguez vers cet onglet

### 💬 **Interaction avec le Chatbot**

#### **Questions Suggérées**
Le chatbot propose **10 questions fréquentes** :
- "C'est quoi une PCA dans GIM ?"
- "Qui peut clôturer un incident GIM ?"
- "Que signifie un statut rouge dans GIM ?"
- "Comment passer d'une ICA à une PCA ?"
- Et plus...

#### **Questions Personnalisées**
1. Tapez votre question dans la zone de texte
2. Cliquez sur **"🚀 Envoyer"**
3. La réponse apparaît instantanément

#### **Historique de Conversation**
- Les 5 derniers échanges sont affichés
- Historique complet sauvegardé (20 échanges max)
- Bouton **"🗑️ Effacer l'historique"** disponible

---

## 📝 Exemples de Questions

### 🎨 **Statuts et Couleurs**
```
"Que signifient les couleurs dans GIM ?"
"Pourquoi mon GIM est passé de vert à rouge ?"
"C'est quoi un statut bleu ?"
```

### 🔄 **Processus et Workflow**
```
"Comment créer un nouveau GIM ?"
"Différence entre ICA et PCA ?"
"Qu'est-ce qu'un Clean Point ?"
"Comment clôturer un incident ?"
```

### 👥 **Rôles et Permissions**
```
"Qui peut modifier un GIM ?"
"Rôle d'un Issue Manager ?"
"Qui valide les PCA ?"
```

### 🔗 **Intégrations Système**
```
"Comment GIM interagit avec Palantir ?"
"Qu'est-ce que eSupplier ?"
"Intégration avec CQI ?"
```

---

## 🎭 Réponses de Démonstration

En **mode démo** (sans API), le chatbot fournit des réponses prédéfinies pour :

### 🔧 **PCA (Permanent Corrective Action)**
```
🔧 PCA (Permanent Corrective Action) dans GIM :

La PCA est l'action corrective permanente qui résout 
définitivement la cause racine d'un problème.

📋 Caractéristiques :
- ✅ Solution définitive (pas temporaire comme l'ICA)
- ✅ Élimine la cause racine identifiée dans la RCA
- ✅ Fait passer le GIM au statut 🟢 Vert clair
- ✅ Doit être validée avant le Clean Point
```

### 🎨 **Statuts et Couleurs**
```
🎨 Statuts et couleurs dans GIM :

- ⚪ Gris : Brouillon (en cours de création)
- 🔴 Rouge : Investigation en cours
- 🟡 Jaune : Root cause connue / ICA en place
- 🟢 Vert clair : PCA identifiée et validée
- 🔵 Bleu : Clean Point validé, incident clôturé
- ⚫ Noir : Annulé / Rejeté

🔄 Flux normal : Gris → Rouge → Jaune → Vert → Bleu
```

---

## 🔍 Interface Utilisateur

### 📱 **Layout Responsive**
- **Sidebar** : Widget compact avec questions rapides
- **Onglet dédié** : Interface complète de chat
- **CSS personnalisé** : Design cohérent avec l'app PCA

### 🎨 **Éléments Visuels**
- **Boîtes de chat** : Messages utilisateur (bleu) et bot (vert)
- **Boutons interactifs** : Questions suggérées cliquables
- **Indicateurs de statut** : API connectée/démo
- **Historique expandable** : Conversations précédentes

---

## 🛠️ Architecture Technique

### 📁 **Fichiers Ajoutés**
```
├── gim_chatbot.py          # Module principal du chatbot
├── config.py               # Configuration centralisée
├── test_gim_integration.py # Tests d'intégration
├── .env.example            # Template de configuration
└── GUIDE_CHATBOT_GIM.md    # Ce guide
```

### 🔧 **Modifications Existantes**
```
├── app.py                  # Interface Streamlit enrichie
├── requirements.txt        # Nouvelles dépendances
└── README.md              # Documentation mise à jour
```

### 🧩 **Classes Principales**
- **`GIMChatbot`** : Logique du chatbot et API
- **`PCAStreamlitApp`** : Interface enrichie avec GIM
- **`Config`** : Configuration centralisée

---

## 🚨 Dépannage

### ❌ **"Chatbot GIM non disponible"**
**Cause** : Erreur de configuration API
**Solution** :
1. Vérifiez la clé API dans `.env`
2. Testez avec `python test_gim_integration.py`
3. Le mode démo reste fonctionnel

### ❌ **"Import dotenv could not be resolved"**
**Cause** : Dépendance manquante
**Solution** :
```bash
pip install python-dotenv
```

### ❌ **"Internal Server Error"**
**Cause** : Problème avec l'API Generative Engine
**Solution** :
1. Vérifiez la connectivité réseau
2. Validez la clé API
3. Le mode démo prend le relais automatiquement

---

## 📈 Logs et Monitoring

### 📄 **Fichier de Log**
- **Emplacement** : `output.log`
- **Format** : Timestamp + Level + Message
- **Contenu** : Questions utilisateur + réponses bot

### 🔍 **Exemple de Log**
```
2025-07-26 17:30:15 - INFO - GIM Chatbot - User: C'est quoi une PCA ?
2025-07-26 17:30:16 - INFO - GIM Chatbot - Bot: PCA (Permanent Corrective Action)...
```

---

## 🎉 Avantages de l'Intégration

### ✅ **Pour les Utilisateurs**
- **Assistance contextuelle** directement dans l'app
- **Pas de changement d'interface** - tout intégré
- **Réponses instantanées** aux questions GIM
- **Apprentissage progressif** de la plateforme

### ✅ **Pour les Développeurs**
- **Architecture modulaire** facilement extensible
- **Mode démo** pour développement sans API
- **Tests automatisés** pour validation
- **Configuration flexible** via variables d'environnement

### ✅ **Pour l'Organisation**
- **Formation intégrée** aux outils existants
- **Réduction du support** grâce à l'auto-assistance
- **Cohérence des réponses** sur GIM
- **Traçabilité** des questions via logs

---

## 🔮 Évolutions Futures

### 🚀 **Améliorations Prévues**
- **Recherche dans la documentation** GIM
- **Intégration avec la base de connaissances**
- **Suggestions proactives** basées sur le contexte
- **Support multi-langues** (FR/EN)

### 🔧 **Extensions Possibles**
- **Chatbot pour d'autres plateformes** (Palantir, CQI)
- **Assistant vocal** avec reconnaissance vocale
- **Intégration Teams/Slack** pour support distribué
- **Analytics avancés** des questions utilisateurs

---

**🎯 Le chatbot GIM est maintenant prêt à assister vos équipes dans l'utilisation de la plateforme Global Issue Management !**
