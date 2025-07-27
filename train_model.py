"""
Module d'entraînement pour le projet NLP de prédiction de solutions techniques (PCA)
Utilise TF-IDF pour la vectorisation et RandomForestClassifier pour la classification
Auteur: Assistant IA
Date: 2025-07-26
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import TextPreprocessor


class PCAPredictionModel:
    """
    Classe pour l'entraînement du modèle de prédiction de PCA
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialise le modèle avec les paramètres par défaut
        
        Args:
            random_state (int): Graine pour la reproductibilité
        """
        self.random_state = random_state
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        self.is_trained = False
        
        # Paramètres TF-IDF
        self.tfidf_params = {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'stop_words': None  # Nous avons déjà supprimé les stop words
        }
        
        # Paramètres RandomForest
        self.rf_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': self.random_state,
            'n_jobs': -1
        }
    
    def create_vectorizer(self) -> TfidfVectorizer:
        """
        Crée et configure le vectoriseur TF-IDF
        
        Returns:
            TfidfVectorizer: Vectoriseur configuré
        """
        return TfidfVectorizer(**self.tfidf_params)
    
    def create_classifier(self) -> RandomForestClassifier:
        """
        Crée et configure le classificateur RandomForest
        
        Returns:
            RandomForestClassifier: Classificateur configuré
        """
        return RandomForestClassifier(**self.rf_params)
    
    def prepare_data(self, X: pd.Series, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Prépare les données pour l'entraînement
        
        Args:
            X (pd.Series): Textes d'entrée
            y (pd.Series): Labels de sortie
            test_size (float): Proportion des données de test
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Encodage des labels si nécessaire
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        print(f"Données d'entraînement: {len(X_train)} exemples")
        print(f"Données de test: {len(X_test)} exemples")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.Series, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Entraîne le modèle complet (vectorisation + classification)
        
        Args:
            X_train (pd.Series): Textes d'entraînement
            y_train (np.ndarray): Labels d'entraînement
            
        Returns:
            Dict[str, Any]: Métriques d'entraînement
        """
        print("=== DÉBUT DE L'ENTRAÎNEMENT ===")
        
        # 1. Vectorisation TF-IDF
        print("Vectorisation TF-IDF...")
        self.vectorizer = self.create_vectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        print(f"Matrice TF-IDF: {X_train_tfidf.shape}")
        print(f"Vocabulaire: {len(self.vectorizer.vocabulary_)} mots")
        
        # 2. Entraînement du classificateur
        print("Entraînement du RandomForestClassifier...")
        self.model = self.create_classifier()
        self.model.fit(X_train_tfidf, y_train)
        
        # 3. Validation croisée
        print("Validation croisée...")
        cv_scores = cross_val_score(
            self.model, X_train_tfidf, y_train, 
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        self.is_trained = True
        
        # Métriques d'entraînement
        train_metrics = {
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'tfidf_shape': X_train_tfidf.shape
        }
        
        print(f"Accuracy moyenne (CV): {train_metrics['cv_mean_accuracy']:.4f} (+/- {train_metrics['cv_std_accuracy']*2:.4f})")
        
        return train_metrics
    
    def evaluate(self, X_test: pd.Series, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Évalue le modèle sur les données de test
        
        Args:
            X_test (pd.Series): Textes de test
            y_test (np.ndarray): Labels de test
            
        Returns:
            Dict[str, Any]: Métriques d'évaluation
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant l'évaluation")
        
        print("=== ÉVALUATION DU MODÈLE ===")
        
        # Vectorisation des données de test
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Prédictions
        y_pred = self.model.predict(X_test_tfidf)
        y_pred_proba = self.model.predict_proba(X_test_tfidf)
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        
        # Conversion des labels pour l'affichage
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Rapport de classification
        class_report = classification_report(
            y_test_labels, y_pred_labels, 
            output_dict=True, zero_division=0
        )
        
        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy sur le test: {accuracy:.4f}")
        print("\nRapport de classification:")
        print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
        
        evaluation_metrics = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred_labels.tolist(),
            'true_labels': y_test_labels.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }
        
        return evaluation_metrics
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Récupère l'importance des features du modèle
        
        Args:
            top_n (int): Nombre de features les plus importantes à retourner
            
        Returns:
            Dict[str, float]: Dictionnaire des features et leur importance
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")
        
        # Récupération des noms des features
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Importance des features
        importances = self.model.feature_importances_
        
        # Création du dictionnaire feature -> importance
        feature_importance = dict(zip(feature_names, importances))
        
        # Tri par importance décroissante
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:top_n])
    
    def save_model(self, model_dir: str = 'models') -> None:
        """
        Sauvegarde le modèle et le vectoriseur
        
        Args:
            model_dir (str): Répertoire de sauvegarde
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant la sauvegarde")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Sauvegarde du modèle
        model_path = os.path.join(model_dir, 'model.pkl')
        joblib.dump(self.model, model_path)
        
        # Sauvegarde du vectoriseur
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Sauvegarde de l'encodeur de labels
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, label_encoder_path)
        
        print(f"Modèle sauvegardé dans {model_dir}/")
        print(f"- {model_path}")
        print(f"- {vectorizer_path}")
        print(f"- {label_encoder_path}")
    
    def load_model(self, model_dir: str = 'models') -> None:
        """
        Charge un modèle sauvegardé
        
        Args:
            model_dir (str): Répertoire contenant le modèle
        """
        model_path = os.path.join(model_dir, 'model.pkl')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        
        if not all(os.path.exists(path) for path in [model_path, vectorizer_path, label_encoder_path]):
            raise FileNotFoundError("Fichiers du modèle manquants")
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.is_trained = True
        
        print(f"Modèle chargé depuis {model_dir}/")


def main():
    """Fonction principale pour l'entraînement"""
    # Initialisation
    preprocessor = TextPreprocessor()
    model = PCAPredictionModel()
    
    try:
        # 1. Chargement et préprocessing des données
        print("=== CHARGEMENT DES DONNÉES ===")
        df, X, y = preprocessor.load_and_preprocess_data('data/gim_diagnostic_dataset.csv')
        
        # 2. Préparation des données
        print("\n=== PRÉPARATION DES DONNÉES ===")
        X_train, X_test, y_train, y_test = model.prepare_data(X, y)
        
        # 3. Entraînement
        print("\n=== ENTRAÎNEMENT ===")
        train_metrics = model.train(X_train, y_train)
        
        # 4. Évaluation
        print("\n=== ÉVALUATION ===")
        eval_metrics = model.evaluate(X_test, y_test)
        
        # 5. Features importantes
        print("\n=== FEATURES IMPORTANTES ===")
        important_features = model.get_feature_importance(top_n=10)
        for feature, importance in important_features.items():
            print(f"{feature}: {importance:.4f}")
        
        # 6. Sauvegarde
        print("\n=== SAUVEGARDE ===")
        model.save_model()
        
        print("\n=== ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ===")
        
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {e}")
        raise


if __name__ == "__main__":
    main()
