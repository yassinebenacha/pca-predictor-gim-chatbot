#!/usr/bin/env python3
"""
Script de configuration des modèles pour le déploiement
Crée les modèles de base si ils n'existent pas
Support pour RandomForest et DistilBERT depuis Hugging Face
"""

import os
import pickle
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Import conditionnel pour transformers
try:
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def create_basic_models():
    """Crée des modèles de base pour le déploiement"""

    # Créer le dossier models s'il n'existe pas
    os.makedirs("models", exist_ok=True)

    # Données d'exemple pour créer les modèles
    sample_data = [
        "P0420 Catalyst System Efficiency Below Threshold",
        "P0171 System Too Lean",
        "P0300 Random/Multiple Cylinder Misfire Detected",
        "P0442 Evaporative Emission Control System Leak Detected",
        "P0128 Coolant Thermostat"
    ]

    sample_labels = [
        "Replace Catalytic Converter",
        "Check Air Filter and MAF Sensor",
        "Replace Spark Plugs",
        "Check EVAP System",
        "Replace Thermostat"
    ]

    # Créer et sauvegarder le vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(sample_data)

    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Créer et sauvegarder le label encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(sample_labels)

    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Créer et sauvegarder le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Modèles RandomForest de base créés avec succès")

def check_distilbert_model():
    """Vérifie si le modèle DistilBERT est disponible et fonctionnel"""

    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers non disponible. Installez avec: pip install transformers torch")
        return False

    distilbert_dir = "distilbert_pca_model"

    # Vérifier les fichiers requis
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.txt",
        "label_mapping.json"
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(distilbert_dir, file)):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Fichiers DistilBERT manquants: {', '.join(missing_files)}")
        return False

    try:
        # Test de chargement
        tokenizer = DistilBertTokenizer.from_pretrained(distilbert_dir)
        model = DistilBertForSequenceClassification.from_pretrained(distilbert_dir)

        # Vérifier le label mapping
        with open(os.path.join(distilbert_dir, "label_mapping.json"), 'r') as f:
            label_mapping = json.load(f)

        print(f"✅ Modèle DistilBERT disponible avec {len(label_mapping)} classes")
        print(f"✅ Modèle configuré pour {model.config.num_labels} labels")

        return True

    except Exception as e:
        print(f"❌ Erreur lors du test DistilBERT: {e}")
        return False

def setup_deployment():
    """Configure les modèles pour le déploiement"""
    print("🚀 Configuration des modèles pour le déploiement...")

    # Créer les modèles RandomForest de base
    create_basic_models()

    # Vérifier DistilBERT
    distilbert_available = check_distilbert_model()

    print("\n📊 Résumé de la configuration:")
    print("✅ RandomForest: Disponible")
    print(f"{'✅' if distilbert_available else '❌'} DistilBERT: {'Disponible' if distilbert_available else 'Non disponible'}")

    if distilbert_available:
        print("\n🎯 Recommandation: Utilisez DistilBERT pour de meilleures performances")
    else:
        print("\n🎯 Recommandation: Utilisez RandomForest (plus rapide, fonctionne hors ligne)")

    return distilbert_available

if __name__ == "__main__":
    setup_deployment()
