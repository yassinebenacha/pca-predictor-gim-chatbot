"""
Test de configuration API pour le chatbot GIM
"""

import os
from dotenv import load_dotenv

# Charger le fichier .env
load_dotenv()

print("🔧 TEST DE CONFIGURATION API")
print("-" * 40)

# Vérifier la clé API
api_key = os.getenv("GENERATIVE_ENGINE_API_KEY")
print(f"Clé API trouvée: {'✅ Oui' if api_key else '❌ Non'}")

if api_key:
    print(f"Clé API: {api_key[:10]}...{api_key[-10:] if len(api_key) > 20 else api_key}")
else:
    print("❌ Clé API non trouvée dans les variables d'environnement")

# Test du chatbot
try:
    from gim_chatbot import GIMChatbot
    chatbot = GIMChatbot()
    
    print(f"Chatbot disponible: {'✅ Oui' if chatbot.is_available else '❌ Non'}")
    
    if chatbot.is_available:
        print("🧪 Test d'une question...")
        response = chatbot.get_response("C'est quoi GIM ?")
        print(f"Réponse reçue: {'✅ Oui' if response and 'Erreur' not in response else '❌ Non'}")
        if response:
            print(f"Début de réponse: {response[:100]}...")
    else:
        print("⚠️ Chatbot en mode démo")
        
except Exception as e:
    print(f"❌ Erreur: {e}")

print("\n💡 Si le chatbot n'est pas disponible:")
print("1. Vérifiez que le fichier .env existe (pas .env.example)")
print("2. Redémarrez l'application Streamlit")
print("3. Vérifiez la validité de la clé API")
