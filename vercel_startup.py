#!/usr/bin/env python3
"""
Script de démarrage pour Vercel
Configure les modèles selon l'environnement
"""

import os
import sys
import subprocess

def install_requirements():
    """Installe les dépendances"""
    print("📦 Installation des dépendances...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dépendances installées")
        return True
    except Exception as e:
        print(f"❌ Erreur installation: {e}")
        return False

def setup_models():
    """Configure les modèles pour Vercel"""
    print("🔧 Configuration des modèles pour Vercel...")
    
    # Vérifier si on est sur Vercel
    is_vercel = os.environ.get('VERCEL', False) or os.environ.get('VERCEL_ENV', False)
    hf_repo = os.environ.get('HF_REPO', None)
    
    if is_vercel:
        print("🌐 Environnement Vercel détecté")
        
        if hf_repo:
            print(f"🤗 Repository Hugging Face configuré: {hf_repo}")
            # Tester la connexion HF
            try:
                from transformers import DistilBertTokenizer
                tokenizer = DistilBertTokenizer.from_pretrained(hf_repo)
                print("✅ Modèle DistilBERT accessible depuis HF Hub")
            except Exception as e:
                print(f"⚠️ Erreur accès HF Hub: {e}")
                print("🔄 Fallback vers RandomForest")
        else:
            print("⚠️ HF_REPO non configuré, utilisation de RandomForest uniquement")
    
    # Créer les modèles de base
    try:
        from setup_models import create_basic_models
        create_basic_models()
        print("✅ Modèles de base créés")
    except Exception as e:
        print(f"❌ Erreur création modèles: {e}")
        return False
    
    return True

def main():
    """Fonction principale"""
    print("🚀 Démarrage Vercel - Configuration PCA Predictor")
    
    # Installation des dépendances
    if not install_requirements():
        sys.exit(1)
    
    # Configuration des modèles
    if not setup_models():
        sys.exit(1)
    
    print("✅ Configuration Vercel terminée avec succès!")
    
    # Afficher les informations de configuration
    print("\n📊 Configuration actuelle:")
    print(f"VERCEL: {os.environ.get('VERCEL', 'Non')}")
    print(f"VERCEL_ENV: {os.environ.get('VERCEL_ENV', 'Non défini')}")
    print(f"HF_REPO: {os.environ.get('HF_REPO', 'Non configuré')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Non défini')}")

if __name__ == "__main__":
    main()
