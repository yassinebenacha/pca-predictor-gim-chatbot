"""
Application Streamlit pour le projet NLP de prédiction de solutions techniques (PCA)
Interface utilisateur pour les ingénieurs non techniques
Auteur: Assistant IA
Date: 2025-07-26
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import json
from typing import Dict, List

from predict import PCAPredictor
from main import PCAMLPipeline
from gim_chatbot import GIMChatbot, demo_mode_response


# Configuration de la page
st.set_page_config(
    page_title="Prédicteur PCA & Assistant GIM",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .gim-chat-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .chat-message-user {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: right;
    }
    .chat-message-bot {
        background-color: #f1f8e9;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class PCAStreamlitApp:
    """
    Application Streamlit pour la prédiction de PCA avec chatbot GIM intégré
    """

    def __init__(self):
        """Initialise l'application"""
        self.predictor = None
        self.pipeline = PCAMLPipeline()
        self.gim_chatbot = GIMChatbot()

        # Configuration par défaut
        self.backend = "randomforest"  # Par défaut
        self.hf_repo = None

        # Initialisation du state
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []

        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False

        if 'show_gim_chat' not in st.session_state:
            st.session_state.show_gim_chat = False

        if 'quick_gim_question' not in st.session_state:
            st.session_state.quick_gim_question = None
    
    def load_model(self, backend: str = None, hf_repo: str = None) -> bool:
        """
        Charge le modèle de prédiction

        Args:
            backend (str): Type de modèle ('randomforest' ou 'distilbert')
            hf_repo (str): Repository Hugging Face pour DistilBERT (optionnel)

        Returns:
            bool: True si le modèle est chargé avec succès
        """
        if backend:
            self.backend = backend
        if hf_repo:
            self.hf_repo = hf_repo

        try:
            cache_key = f"{self.backend}_{self.hf_repo or 'local'}"
            if not st.session_state.get(f"model_loaded_{cache_key}", False):
                with st.spinner(f"Chargement du modèle {self.backend.upper()}..."):
                    self.predictor = PCAPredictor(
                        backend=self.backend,
                        checkpoint_dir="distilbert_pca_model",
                        hf_repo=self.hf_repo
                    )
                    self.predictor.load_model()
                    st.session_state[f"model_loaded_{cache_key}"] = True
                st.success(f"✅ Modèle {self.backend.upper()} chargé avec succès!")
            else:
                self.predictor = PCAPredictor(
                    backend=self.backend,
                    checkpoint_dir="distilbert_pca_model",
                    hf_repo=self.hf_repo
                )
                self.predictor.load_model()
            return True
        except Exception as e:
            # Fallback vers RandomForest si DistilBERT échoue
            if self.backend == "distilbert":
                st.warning("⚠️ Modèle DistilBERT non disponible, utilisation du modèle RandomForest")
                return self.load_model("randomforest")
            else:
                st.error(f"❌ Erreur lors du chargement du modèle: {e}")
                st.info("💡 Assurez-vous d'avoir entraîné le modèle avec `python main.py --action train`")
                return False
    
    def render_header(self):
        """Affiche l'en-tête de l'application"""
        st.markdown('<h1 class="main-header">🔧 Prédicteur PCA & Assistant GIM</h1>',
                   unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Système IA pour prédire automatiquement les solutions techniques (PCA)
                à partir de codes DTC et descriptions de pannes
            </p>
            <p style="font-size: 1rem; color: #888;">
                🤖 <strong>Nouveau :</strong> Assistant GIM intégré pour vos questions sur la plateforme Global Issue Management
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Affiche la barre latérale avec les informations du modèle"""
        st.sidebar.header("🤖 Configuration du Modèle")

        # Choix du backend
        backend_options = {
            "randomforest": "🌲 RandomForest (Rapide)",
            "distilbert": "🧠 DistilBERT (Précis)"
        }

        selected_backend = st.sidebar.selectbox(
            "Type de modèle",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x],
            index=0 if self.backend == "randomforest" else 1,
            help="RandomForest: Plus rapide, fonctionne hors ligne\nDistilBERT: Plus précis, nécessite plus de ressources"
        )

        # Configuration DistilBERT
        hf_repo = None
        if selected_backend == "distilbert":
            st.sidebar.subheader("⚙️ Configuration DistilBERT")
            use_hf = st.sidebar.checkbox(
                "Utiliser Hugging Face Hub",
                help="Télécharge le modèle depuis Hugging Face si le modèle local n'est pas disponible"
            )

            if use_hf:
                hf_repo = st.sidebar.text_input(
                    "Repository HF (optionnel)",
                    placeholder="ex: votre-username/pca-distilbert",
                    help="Laissez vide pour utiliser le modèle par défaut"
                )

        # Bouton pour charger/recharger le modèle
        if st.sidebar.button("🔄 Charger le modèle", use_container_width=True):
            self.load_model(selected_backend, hf_repo)

        # Mise à jour automatique si le backend change
        if selected_backend != self.backend:
            self.backend = selected_backend
            self.hf_repo = hf_repo

        st.sidebar.markdown("---")
        st.sidebar.header("📊 Informations du Modèle")

        # Informations sur le modèle
        model_info = self.pipeline.get_model_info()
        
        if model_info['is_trained']:
            st.sidebar.success("✅ Modèle disponible")
            
            # Affichage des détails
            st.sidebar.subheader("📁 Fichiers du modèle")
            for file, status in model_info['model_files'].items():
                if status['exists']:
                    st.sidebar.text(f"✅ {file}")
                    if status['modified']:
                        modified_date = datetime.fromisoformat(status['modified']).strftime("%d/%m/%Y %H:%M")
                        st.sidebar.caption(f"   Modifié: {modified_date}")
                else:
                    st.sidebar.text(f"❌ {file}")
        else:
            st.sidebar.error("❌ Modèle non disponible")
            st.sidebar.info("Entraînez d'abord le modèle avec la commande:\n`python main.py --action train`")
        
        # Historique des prédictions
        st.sidebar.subheader("📈 Historique")
        if st.session_state.prediction_history:
            st.sidebar.text(f"Prédictions effectuées: {len(st.session_state.prediction_history)}")

            if st.sidebar.button("🗑️ Effacer l'historique"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.sidebar.text("Aucune prédiction encore")

        # Widget chatbot GIM dans la sidebar
        self.gim_chatbot.render_sidebar_widget()
    
    def render_prediction_form(self):
        """Affiche le formulaire de prédiction"""
        st.header("🎯 Nouvelle Prédiction")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                code_dtc = st.text_input(
                    "Code DTC",
                    placeholder="Ex: P0300, P0171, B1234...",
                    help="Code de diagnostic de panne (Diagnostic Trouble Code)"
                )
                
                description = st.text_area(
                    "Description du problème",
                    placeholder="Ex: Engine misfiring randomly, Check engine light on...",
                    height=100,
                    help="Description détaillée du problème observé"
                )
            
            with col2:
                root_cause = st.text_area(
                    "Cause racine (optionnel)",
                    placeholder="Ex: Faulty spark plugs, Vacuum leak...",
                    height=100,
                    help="Description de la cause racine si connue"
                )
                
                # Options avancées
                st.subheader("⚙️ Options")
                show_probabilities = st.checkbox("Afficher toutes les probabilités", value=True)
                top_n = st.slider("Nombre de prédictions alternatives", 1, 5, 3)
            
            # Bouton de prédiction
            predict_button = st.form_submit_button("🔮 Prédire la PCA", use_container_width=True)
            
            if predict_button:
                if not code_dtc.strip() or not description.strip():
                    st.error("❌ Veuillez remplir au minimum le Code DTC et la Description du problème")
                else:
                    self.make_prediction(code_dtc, description, root_cause, show_probabilities, top_n)
    
    def make_prediction(self, code_dtc: str, description: str, root_cause: str, 
                       show_probabilities: bool, top_n: int):
        """
        Effectue une prédiction et affiche les résultats
        
        Args:
            code_dtc (str): Code DTC
            description (str): Description du problème
            root_cause (str): Cause racine
            show_probabilities (bool): Afficher les probabilités
            top_n (int): Nombre de prédictions alternatives
        """
        if not self.load_model():
            return
        
        with st.spinner("Analyse en cours..."):
            try:
                # Prédiction principale
                result = self.predictor.predict_single(
                    code_dtc, description, root_cause, 
                    return_probabilities=show_probabilities
                )
                
                if 'error' in result:
                    st.error(f"❌ Erreur lors de la prédiction: {result['error']}")
                    return
                
                # Affichage du résultat principal
                self.display_main_prediction(result)
                
                # Prédictions alternatives
                if show_probabilities and top_n > 1:
                    top_predictions = self.predictor.get_top_predictions(
                        code_dtc, description, root_cause, top_n
                    )
                    self.display_alternative_predictions(top_predictions)
                
                # Explication détaillée
                explanation = self.predictor.explain_prediction(code_dtc, description, root_cause)
                self.display_explanation(explanation)
                
                # Sauvegarde dans l'historique
                self.save_to_history(result, code_dtc, description, root_cause)
                
            except Exception as e:
                st.error(f"❌ Erreur inattendue: {e}")
    
    def display_main_prediction(self, result: Dict):
        """Affiche le résultat principal de la prédiction"""
        st.header("🎯 Résultat de la Prédiction")
        
        # Détermination de la couleur selon la confiance
        confidence = result.get('confidence', 0)
        if confidence >= 0.7:
            confidence_class = "confidence-high"
            confidence_icon = "🟢"
        elif confidence >= 0.4:
            confidence_class = "confidence-medium"
            confidence_icon = "🟡"
        else:
            confidence_class = "confidence-low"
            confidence_icon = "🔴"
        
        # Affichage dans une boîte stylée
        st.markdown(f"""
        <div class="prediction-box">
            <h3>🔧 PCA Recommandée</h3>
            <h2 style="color: #1f77b4; margin: 1rem 0;">{result['predicted_pca']}</h2>
            <p><strong>Niveau de confiance:</strong> 
               <span class="{confidence_class}">{confidence_icon} {confidence:.1%}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Métriques en colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confiance", f"{confidence:.1%}")
        
        with col2:
            confidence_level = self.get_confidence_level(confidence)
            st.metric("Niveau", confidence_level)
        
        with col3:
            st.metric("Texte traité", f"{len(result['processed_text'].split())} mots")
    
    def display_alternative_predictions(self, top_predictions: List[Dict]):
        """Affiche les prédictions alternatives"""
        st.header("🔄 Solutions Alternatives")
        
        # Création d'un graphique en barres
        if len(top_predictions) > 1:
            df_alternatives = pd.DataFrame([
                {
                    'PCA': pred['pca'][:50] + '...' if len(pred['pca']) > 50 else pred['pca'],
                    'Probabilité': pred['probability'],
                    'Confiance': pred['confidence_level']
                }
                for pred in top_predictions[1:]  # Exclure la première (déjà affichée)
            ])
            
            fig = px.bar(
                df_alternatives, 
                x='Probabilité', 
                y='PCA',
                orientation='h',
                title="Probabilités des Solutions Alternatives",
                color='Probabilité',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé
        for i, pred in enumerate(top_predictions[1:], 2):
            with st.expander(f"#{i} - {pred['pca'][:60]}... (Prob: {pred['probability']:.1%})"):
                st.write(f"**PCA complète:** {pred['pca']}")
                st.write(f"**Probabilité:** {pred['probability']:.3f}")
                st.write(f"**Niveau de confiance:** {pred['confidence_level']}")
    
    def display_explanation(self, explanation: Dict):
        """Affiche l'explication détaillée de la prédiction"""
        with st.expander("🔍 Explication Détaillée", expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📝 Analyse de l'entrée")
                st.write(f"**Texte original:**")
                st.code(f"DTC: {explanation['input_analysis']['original_input']['code_dtc']}")
                st.code(f"Description: {explanation['input_analysis']['original_input']['description']}")
                if explanation['input_analysis']['original_input']['root_cause']:
                    st.code(f"Cause: {explanation['input_analysis']['original_input']['root_cause']}")
                
                st.write(f"**Texte préprocessé:**")
                st.code(explanation['input_analysis']['processed_text'])
                
                st.write(f"**Statistiques:**")
                st.write(f"- Longueur: {explanation['input_analysis']['text_length']} caractères")
                st.write(f"- Mots: {explanation['input_analysis']['word_count']} mots")
            
            with col2:
                st.subheader("🎯 Résumé de la prédiction")
                summary = explanation['prediction_summary']
                st.write(f"**PCA prédite:** {summary['predicted_pca']}")
                st.write(f"**Confiance:** {summary['confidence']:.3f}")
                st.write(f"**Niveau:** {summary['confidence_level']}")
                
                if explanation['alternative_solutions']:
                    st.write("**Autres solutions possibles:**")
                    for alt in explanation['alternative_solutions']:
                        st.write(f"- {alt['pca'][:40]}... ({alt['probability']:.1%})")
    
    def save_to_history(self, result: Dict, code_dtc: str, description: str, root_cause: str):
        """Sauvegarde la prédiction dans l'historique"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': {
                'code_dtc': code_dtc,
                'description': description,
                'root_cause': root_cause
            },
            'prediction': result['predicted_pca'],
            'confidence': result.get('confidence', 0)
        }
        
        st.session_state.prediction_history.append(history_entry)
        
        # Limiter l'historique à 50 entrées
        if len(st.session_state.prediction_history) > 50:
            st.session_state.prediction_history = st.session_state.prediction_history[-50:]
    
    def render_history(self):
        """Affiche l'historique des prédictions"""
        if not st.session_state.prediction_history:
            st.info("📝 Aucune prédiction dans l'historique")
            return
        
        st.header("📈 Historique des Prédictions")
        
        # Conversion en DataFrame pour l'affichage
        history_data = []
        for entry in reversed(st.session_state.prediction_history[-10:]):  # 10 dernières
            history_data.append({
                'Horodatage': datetime.fromisoformat(entry['timestamp']).strftime("%d/%m/%Y %H:%M"),
                'Code DTC': entry['input']['code_dtc'],
                'Description': entry['input']['description'][:50] + '...' if len(entry['input']['description']) > 50 else entry['input']['description'],
                'PCA Prédite': entry['prediction'][:50] + '...' if len(entry['prediction']) > 50 else entry['prediction'],
                'Confiance': f"{entry['confidence']:.1%}"
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True)
        
        # Graphique de l'évolution de la confiance
        if len(st.session_state.prediction_history) > 1:
            confidences = [entry['confidence'] for entry in st.session_state.prediction_history[-20:]]
            timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in st.session_state.prediction_history[-20:]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=confidences,
                mode='lines+markers',
                name='Confiance',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(
                title="Évolution de la Confiance des Prédictions",
                xaxis_title="Temps",
                yaxis_title="Confiance",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def get_confidence_level(self, confidence: float) -> str:
        """Détermine le niveau de confiance"""
        if confidence >= 0.8:
            return "Très élevée"
        elif confidence >= 0.6:
            return "Élevée"
        elif confidence >= 0.4:
            return "Moyenne"
        elif confidence >= 0.2:
            return "Faible"
        else:
            return "Très faible"
    
    def render_gim_chat_tab(self):
        """Affiche l'onglet du chatbot GIM"""
        st.header("🤖 Assistant GIM - Global Issue Management")

        # Gestion des questions rapides depuis la sidebar
        if st.session_state.quick_gim_question:
            st.info(f"💬 Question rapide : {st.session_state.quick_gim_question}")

            # Traiter la question automatiquement
            if 'gim_chat_history' not in st.session_state:
                st.session_state.gim_chat_history = []

            # Obtenir la réponse
            if self.gim_chatbot.is_available:
                response = self.gim_chatbot.get_response(st.session_state.quick_gim_question)
            else:
                response = demo_mode_response(st.session_state.quick_gim_question)

            # Ajouter à l'historique
            exchange = {
                "user": st.session_state.quick_gim_question,
                "bot": response,
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
            st.session_state.gim_chat_history.append(exchange)

            # Réinitialiser la question rapide
            st.session_state.quick_gim_question = None

        # Interface de chat complète
        self.gim_chatbot.render_chat_interface()

    def run(self):
        """Lance l'application Streamlit"""
        self.render_header()
        self.render_sidebar()

        # Gestion de l'affichage du chat GIM
        if st.session_state.show_gim_chat:
            # Onglets avec GIM visible
            tab1, tab2, tab3 = st.tabs(["🎯 Prédiction PCA", "🤖 Assistant GIM", "📈 Historique"])

            with tab1:
                self.render_prediction_form()

            with tab2:
                self.render_gim_chat_tab()

            with tab3:
                self.render_history()
        else:
            # Onglets normaux
            tab1, tab2 = st.tabs(["🎯 Prédiction PCA", "📈 Historique"])

            with tab1:
                self.render_prediction_form()

                # Bouton pour ouvrir le chat GIM
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🤖 Ouvrir l'Assistant GIM", use_container_width=True, type="secondary"):
                        st.session_state.show_gim_chat = True
                        st.rerun()

            with tab2:
                self.render_history()


def main():
    """Fonction principale de l'application Streamlit"""
    app = PCAStreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
