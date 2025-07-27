"""
Configuration pour l'application PCA + GIM Chatbot
Auteur: Assistant IA
Date: 2025-07-26
"""

import os
from typing import Optional


class Config:
    """
    Configuration centralisée pour l'application
    """
    
    # Configuration API Generative Engine
    GENERATIVE_ENGINE_API_KEY: Optional[str] = os.getenv("GENERATIVE_ENGINE_API_KEY")
    GENERATIVE_ENGINE_BASE_URL: str = "https://openai.generative.engine.capgemini.com/v1"
    GENERATIVE_ENGINE_MODEL: str = "openai.gpt-4o"
    
    # Configuration application
    APP_TITLE: str = "Prédicteur PCA & Assistant GIM"
    APP_ICON: str = "🔧"
    
    # Configuration logging
    LOG_FILE: str = "output.log"
    LOG_LEVEL: str = "INFO"
    
    # Configuration modèles ML
    MODEL_DIR: str = "models"
    DATA_DIR: str = "data"
    
    # Configuration chatbot
    MAX_CHAT_HISTORY: int = 20
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.7
    
    @classmethod
    def is_api_configured(cls) -> bool:
        """
        Vérifie si l'API Generative Engine est configurée
        
        Returns:
            bool: True si configurée
        """
        return cls.GENERATIVE_ENGINE_API_KEY is not None and cls.GENERATIVE_ENGINE_API_KEY != ""
    
    @classmethod
    def get_api_status(cls) -> str:
        """
        Retourne le statut de configuration de l'API
        
        Returns:
            str: Message de statut
        """
        if cls.is_api_configured():
            return "✅ API Generative Engine configurée"
        else:
            return "⚠️ API Generative Engine non configurée - Mode démo actif"


# Variables d'environnement pour le développement
def setup_dev_environment():
    """
    Configure l'environnement de développement
    """
    # Exemple de configuration pour le développement local
    # À adapter selon vos besoins
    
    if not os.getenv("GENERATIVE_ENGINE_API_KEY"):
        print("⚠️ Variable d'environnement GENERATIVE_ENGINE_API_KEY non définie")
        print("💡 Pour activer le chatbot GIM, définissez cette variable :")
        print("   export GENERATIVE_ENGINE_API_KEY='votre-clé-api'")
        print("   ou créez un fichier .env avec cette variable")


if __name__ == "__main__":
    setup_dev_environment()
    print(f"Configuration API : {Config.get_api_status()}")
