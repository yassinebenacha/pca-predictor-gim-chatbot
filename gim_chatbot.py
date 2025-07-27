"""
Module Chatbot GIM pour l'assistance sur la plateforme Global Issue Management
Intégré dans l'application Streamlit de prédiction PCA
Auteur: Assistant IA
Date: 2025-07-26
"""

import streamlit as st
from openai import OpenAI
import logging
from datetime import datetime
from typing import List, Dict, Optional
import json
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


class GIMChatbot:
    """
    Chatbot intelligent pour l'assistance sur la plateforme GIM
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialise le chatbot GIM
        
        Args:
            api_key (str): Clé API pour Generative Engine (optionnel)
        """
        self.api_key = api_key or os.getenv("GENERATIVE_ENGINE_API_KEY", "demo-key")
        self.base_url = "https://openai.generative.engine.capgemini.com/v1"
        self.model = "openai.gpt-4o"
        
        # Configuration du logging
        logging.basicConfig(
            filename='output.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialisation du client OpenAI
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.is_available = True
        except Exception as e:
            logging.error(f"Erreur initialisation client OpenAI: {e}")
            self.is_available = False
        
        # Prompt système pour GIM
        self.system_prompt = """Tu es un assistant intelligent dédié à la plateforme **GIM (Global Issue Management)** utilisée pour la gestion des incidents industriels.

Ta mission est d'aider les utilisateurs à **comprendre, utiliser et naviguer dans GIM**, y compris les statuts, les rôles, les étapes du traitement des incidents, et les interactions avec d'autres systèmes comme Palantir, eSupplier, CQI...

Voici les thématiques principales sur lesquelles tu peux répondre :

### 📌 1. Vue d'ensemble du système GIM
- Qu'est-ce que GIM ?
- À quoi sert la plateforme ?
- Qui l'utilise ? (Issue Manager, Business Supervisor, Solving Team…)

### 🎨 2. Signification des statuts et couleurs dans GIM
- Gris : brouillon
- 🔴 Rouge : investigation
- 🟡 Jaune : root cause connue / ICA en place
- 🟢 Vert clair : PCA identifiée
- 🔵 Bleu : clean point validé, clôture
- Noir : annulé / rejeté

### 🧩 3. Sections clés du processus
- ICA (Interim Containment Action)
- RCA (Root Cause Analysis)
- PCA (Permanent Corrective Action)
- Clean Point Date
- Attachments, Stakeholders, Incidents

### 🔄 4. Flux de traitement GIM
- Comment une issue est créée, suivie et clôturée
- Différences entre problème Production vs Développement
- Validation par les rôles (Issue Manager, Business Supervisor, etc.)

### 🔗 5. Interactions systèmes GIM
- Intégration avec Palantir, eSupplier, CQI, DWH, VIN DB...
- Documents joints, assignations, PRAS (Part Return Analysis), Systran (traductions), etc.

**Ton ton est professionnel, clair, pédagogique.**
Tu es là pour **former et assister** les utilisateurs de GIM, qu'ils soient ingénieurs qualité, data analysts ou opérationnels.

Si l'utilisateur pose une question imprécise, demande-lui de reformuler ou propose-lui un exemple.

Réponds de manière concise mais complète, en utilisant des emojis pour structurer tes réponses."""
    
    def get_response(self, user_message: str, conversation_history: List[Dict] = None) -> str:
        """
        Obtient une réponse du chatbot GIM
        
        Args:
            user_message (str): Message de l'utilisateur
            conversation_history (List[Dict]): Historique de conversation
            
        Returns:
            str: Réponse du chatbot
        """
        if not self.is_available:
            return "❌ Le chatbot GIM n'est pas disponible actuellement. Veuillez vérifier la configuration de l'API."
        
        try:
            # Construction des messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Ajout de l'historique si disponible
            if conversation_history:
                messages.extend(conversation_history[-10:])  # Garder les 10 derniers échanges
            
            # Ajout du message utilisateur
            messages.append({"role": "user", "content": user_message})
            
            # Appel à l'API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Extraction de la réponse
            bot_response = response.choices[0].message.content
            
            # Logging
            logging.info(f"GIM Chatbot - User: {user_message[:100]}...")
            logging.info(f"GIM Chatbot - Bot: {bot_response[:100]}...")
            
            return bot_response
            
        except Exception as e:
            error_msg = f"Erreur lors de la communication avec le chatbot: {str(e)}"
            logging.error(error_msg)
            return f"❌ {error_msg}"
    
    def get_suggested_questions(self) -> List[str]:
        """
        Retourne une liste de questions suggérées pour GIM
        
        Returns:
            List[str]: Questions suggérées
        """
        return [
            "C'est quoi une PCA dans GIM ?",
            "Qui peut clôturer un incident GIM ?",
            "Que signifie un statut rouge dans GIM ?",
            "Comment passer d'une ICA à une PCA ?",
            "Pourquoi un GIM repasse de vert à rouge ?",
            "Quels sont les rôles dans GIM ?",
            "Comment GIM interagit avec Palantir ?",
            "Qu'est-ce qu'un Clean Point ?",
            "Différence entre RCA et PCA ?",
            "Comment annuler un GIM ?"
        ]
    
    def render_chat_interface(self):
        """
        Rend l'interface de chat dans Streamlit
        """
        st.header("🤖 Assistant GIM")
        st.markdown("*Votre assistant intelligent pour la plateforme Global Issue Management*")
        
        # Initialisation de l'historique de conversation
        if 'gim_chat_history' not in st.session_state:
            st.session_state.gim_chat_history = []
        
        # Affichage du statut de connexion
        if self.is_available:
            st.success("✅ Chatbot GIM connecté et prêt")
        else:
            st.error("❌ Chatbot GIM non disponible - Mode démo")
        
        # Questions suggérées
        st.subheader("💡 Questions suggérées")
        suggested_questions = self.get_suggested_questions()
        
        # Affichage des questions suggérées en colonnes
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions[:6]):  # Afficher 6 questions
            col = cols[i % 2]
            if col.button(f"❓ {question}", key=f"suggested_{i}"):
                # Ajouter la question à l'historique et obtenir la réponse
                self._process_question(question)
        
        # Zone de saisie
        st.subheader("💬 Posez votre question")
        user_input = st.text_area(
            "Votre question sur GIM :",
            placeholder="Ex: Comment savoir si un problème est résolu dans GIM ?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🚀 Envoyer", type="primary"):
                if user_input.strip():
                    self._process_question(user_input)
                else:
                    st.warning("⚠️ Veuillez saisir une question")
        
        with col2:
            if st.button("🗑️ Effacer l'historique"):
                st.session_state.gim_chat_history = []
                st.rerun()
        
        # Affichage de l'historique de conversation
        if st.session_state.gim_chat_history:
            st.subheader("📝 Historique de conversation")
            
            # Conteneur scrollable pour l'historique
            with st.container():
                for i, exchange in enumerate(reversed(st.session_state.gim_chat_history[-5:])):  # 5 derniers échanges
                    with st.expander(f"💬 Question {len(st.session_state.gim_chat_history) - i}", expanded=(i == 0)):
                        st.markdown(f"**👤 Vous :** {exchange['user']}")
                        st.markdown(f"**🤖 Assistant GIM :** {exchange['bot']}")
                        st.caption(f"⏰ {exchange['timestamp']}")
    
    def _process_question(self, question: str):
        """
        Traite une question et met à jour l'historique
        
        Args:
            question (str): Question de l'utilisateur
        """
        with st.spinner("🤔 L'assistant GIM réfléchit..."):
            # Obtenir la réponse
            response = self.get_response(question, st.session_state.gim_chat_history)
            
            # Ajouter à l'historique
            exchange = {
                "user": question,
                "bot": response,
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
            st.session_state.gim_chat_history.append(exchange)
            
            # Limiter l'historique à 20 échanges
            if len(st.session_state.gim_chat_history) > 20:
                st.session_state.gim_chat_history = st.session_state.gim_chat_history[-20:]
        
        # Recharger la page pour afficher la nouvelle réponse
        st.rerun()
    
    def render_sidebar_widget(self):
        """
        Rend un widget compact dans la sidebar
        """
        st.sidebar.markdown("---")
        st.sidebar.header("🤖 Assistant GIM")
        
        if st.sidebar.button("💬 Ouvrir le Chat GIM", use_container_width=True):
            st.session_state.show_gim_chat = True
        
        # Questions rapides dans la sidebar
        st.sidebar.markdown("**Questions rapides :**")
        quick_questions = [
            "Statuts GIM ?",
            "C'est quoi PCA ?",
            "Qui clôture GIM ?"
        ]
        
        for q in quick_questions:
            if st.sidebar.button(f"❓ {q}", key=f"sidebar_{q}"):
                st.session_state.quick_gim_question = q
                st.session_state.show_gim_chat = True


def demo_mode_response(question: str) -> str:
    """
    Réponses de démonstration quand l'API n'est pas disponible
    
    Args:
        question (str): Question de l'utilisateur
        
    Returns:
        str: Réponse de démonstration
    """
    demo_responses = {
        "pca": """🔧 **PCA (Permanent Corrective Action)** dans GIM :

La PCA est l'**action corrective permanente** qui résout définitivement la cause racine d'un problème.

📋 **Caractéristiques :**
- ✅ Solution définitive (pas temporaire comme l'ICA)
- ✅ Élimine la cause racine identifiée dans la RCA
- ✅ Fait passer le GIM au statut 🟢 **Vert clair**
- ✅ Doit être validée avant le Clean Point

🔄 **Processus :**
1. RCA terminée → Cause racine identifiée
2. PCA définie → Solution permanente proposée
3. PCA validée → Statut vert clair
4. PCA implémentée → Clean Point possible""",
        
        "statut": """🎨 **Statuts et couleurs dans GIM :**

- **⚪ Gris** : Brouillon (en cours de création)
- **🔴 Rouge** : Investigation en cours
- **🟡 Jaune** : Root cause connue / ICA en place
- **🟢 Vert clair** : PCA identifiée et validée
- **🔵 Bleu** : Clean Point validé, incident clôturé
- **⚫ Noir** : Annulé / Rejeté

🔄 **Flux normal :**
Gris → Rouge → Jaune → Vert → Bleu""",
        
        "cloture": """🔒 **Qui peut clôturer un incident GIM ?**

👥 **Rôles autorisés :**
- **Issue Manager** : Responsable principal du GIM
- **Business Supervisor** : Validation métier
- **Quality Manager** : Validation qualité

📋 **Conditions de clôture :**
- ✅ PCA implémentée et validée
- ✅ Clean Point Date atteinte
- ✅ Efficacité de la solution vérifiée
- ✅ Documents de clôture fournis

🔵 **Résultat :** Statut passe au **Bleu** (clôturé)"""
    }
    
    question_lower = question.lower()
    
    if "pca" in question_lower:
        return demo_responses["pca"]
    elif "statut" in question_lower or "couleur" in question_lower:
        return demo_responses["statut"]
    elif "clôtur" in question_lower or "fermer" in question_lower:
        return demo_responses["cloture"]
    else:
        return f"""🤖 **Réponse de démonstration pour :** "{question}"

📌 **Mode démo actif** - L'API Generative Engine n'est pas configurée.

💡 **Exemples de questions que je peux traiter :**
- "C'est quoi une PCA dans GIM ?"
- "Que signifient les statuts colorés ?"
- "Qui peut clôturer un incident ?"
- "Comment passer de rouge à vert ?"
- "Qu'est-ce qu'un Clean Point ?"

🔧 **Pour activer le chatbot complet :**
Configurez la variable d'environnement `GENERATIVE_ENGINE_API_KEY`"""
