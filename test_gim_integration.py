"""
Script de test pour l'intégration du chatbot GIM dans l'application Streamlit
Auteur: Assistant IA
Date: 2025-07-26
"""

import sys
import os

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gim_chatbot import GIMChatbot, demo_mode_response
    from config import Config
    print("✅ Import des modules réussi")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)


def test_config():
    """Test de la configuration"""
    print("\n🔧 TEST DE CONFIGURATION")
    print("-" * 40)
    
    print(f"API configurée: {Config.is_api_configured()}")
    print(f"Statut: {Config.get_api_status()}")
    print(f"Base URL: {Config.GENERATIVE_ENGINE_BASE_URL}")
    print(f"Modèle: {Config.GENERATIVE_ENGINE_MODEL}")


def test_chatbot_init():
    """Test d'initialisation du chatbot"""
    print("\n🤖 TEST D'INITIALISATION DU CHATBOT")
    print("-" * 40)
    
    try:
        chatbot = GIMChatbot()
        print(f"✅ Chatbot initialisé")
        print(f"Disponible: {chatbot.is_available}")
        print(f"Modèle: {chatbot.model}")
        return chatbot
    except Exception as e:
        print(f"❌ Erreur initialisation chatbot: {e}")
        return None


def test_demo_responses():
    """Test des réponses de démonstration"""
    print("\n🎭 TEST DES RÉPONSES DE DÉMONSTRATION")
    print("-" * 40)
    
    test_questions = [
        "C'est quoi une PCA ?",
        "Quels sont les statuts GIM ?",
        "Qui peut clôturer un incident ?",
        "Question générale sur GIM"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        response = demo_mode_response(question)
        print(f"🤖 Réponse: {response[:100]}...")


def test_suggested_questions():
    """Test des questions suggérées"""
    print("\n💡 TEST DES QUESTIONS SUGGÉRÉES")
    print("-" * 40)
    
    chatbot = GIMChatbot()
    questions = chatbot.get_suggested_questions()
    
    print(f"Nombre de questions suggérées: {len(questions)}")
    for i, q in enumerate(questions[:5], 1):
        print(f"{i}. {q}")


def test_api_call():
    """Test d'appel API (si configurée)"""
    print("\n🌐 TEST D'APPEL API")
    print("-" * 40)
    
    chatbot = GIMChatbot()
    
    if not chatbot.is_available:
        print("⚠️ API non disponible - test ignoré")
        return
    
    try:
        response = chatbot.get_response("C'est quoi GIM ?")
        print(f"✅ Appel API réussi")
        print(f"Réponse: {response[:200]}...")
    except Exception as e:
        print(f"❌ Erreur appel API: {e}")


def main():
    """Fonction principale de test"""
    print("🧪 TEST D'INTÉGRATION GIM CHATBOT")
    print("=" * 50)
    
    # Tests
    test_config()
    test_chatbot_init()
    test_demo_responses()
    test_suggested_questions()
    test_api_call()
    
    print("\n" + "=" * 50)
    print("✅ TESTS TERMINÉS")
    print("\n💡 Pour lancer l'application complète:")
    print("   streamlit run app.py")
    print("\n🔧 Pour configurer l'API:")
    print("   1. Copiez .env.example vers .env")
    print("   2. Remplissez GENERATIVE_ENGINE_API_KEY")
    print("   3. Relancez l'application")


if __name__ == "__main__":
    main()
