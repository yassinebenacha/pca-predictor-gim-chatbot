#!/usr/bin/env python3
"""
Script de configuration des modèles pour le déploiement
Crée les modèles de base si ils n'existent pas
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

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
    
    print("✅ Modèles de base créés avec succès")

if __name__ == "__main__":
    create_basic_models()
