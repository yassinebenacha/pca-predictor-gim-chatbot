"""
Module de prédiction pour le projet NLP de prédiction de solutions techniques (PCA)
Permet de faire des prédictions sur de nouveaux exemples
Auteur: Assistant IA
Date: 2025-07-26
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Tuple, Union
from preprocessing import TextPreprocessor


class PCAPredictor:
    """
    Classe pour faire des prédictions de PCA sur de nouveaux exemples
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialise le prédicteur
        
        Args:
            model_dir (str): Répertoire contenant les modèles sauvegardés
        """
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.preprocessor = TextPreprocessor()
        self.is_loaded = False
    
    def load_model(self) -> None:
        """
        Charge le modèle, le vectoriseur et l'encodeur de labels
        """
        model_path = os.path.join(self.model_dir, 'model.pkl')
        vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
        label_encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        
        # Vérification de l'existence des fichiers
        missing_files = []
        for path, name in [(model_path, 'model.pkl'), 
                          (vectorizer_path, 'vectorizer.pkl'),
                          (label_encoder_path, 'label_encoder.pkl')]:
            if not os.path.exists(path):
                missing_files.append(name)
        
        if missing_files:
            raise FileNotFoundError(
                f"Fichiers manquants dans {self.model_dir}: {', '.join(missing_files)}\n"
                f"Veuillez d'abord entraîner le modèle avec train_model.py"
            )
        
        # Chargement des composants
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.label_encoder = joblib.load(label_encoder_path)
            self.is_loaded = True
            print(f"Modèle chargé avec succès depuis {self.model_dir}")
            print(f"Classes disponibles: {list(self.label_encoder.classes_)}")
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")
    
    def preprocess_input(self, code_dtc: str, description: str, root_cause: str = "") -> str:
        """
        Préprocesse un nouvel exemple d'entrée
        
        Args:
            code_dtc (str): Code DTC
            description (str): Description du problème
            root_cause (str): Description de la cause racine (optionnel)
            
        Returns:
            str: Texte préprocessé et concaténé
        """
        # Création d'un DataFrame temporaire pour utiliser le preprocessor
        temp_df = pd.DataFrame({
            'Code DTC': [code_dtc],
            'Description du problème': [description],
            'Root Cause Description': [root_cause if root_cause else ""]
        })
        
        # Préprocessing
        processed_df = self.preprocessor.concatenate_text_columns(temp_df)
        
        return processed_df['texte_concatene'].iloc[0]
    
    def predict_single(self, code_dtc: str, description: str, 
                      root_cause: str = "", return_probabilities: bool = True) -> Dict:
        """
        Fait une prédiction sur un seul exemple
        
        Args:
            code_dtc (str): Code DTC
            description (str): Description du problème
            root_cause (str): Description de la cause racine (optionnel)
            return_probabilities (bool): Si True, retourne les probabilités
            
        Returns:
            Dict: Résultat de la prédiction
        """
        if not self.is_loaded:
            self.load_model()
        
        # Préprocessing de l'entrée
        processed_text = self.preprocess_input(code_dtc, description, root_cause)
        
        if not processed_text.strip():
            return {
                'error': 'Texte vide après préprocessing',
                'processed_text': processed_text
            }
        
        # Vectorisation
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Prédiction
        prediction_encoded = self.model.predict(text_tfidf)[0]
        predicted_pca = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        result = {
            'input': {
                'code_dtc': code_dtc,
                'description': description,
                'root_cause': root_cause
            },
            'processed_text': processed_text,
            'predicted_pca': predicted_pca,
            'confidence': None,
            'all_probabilities': None
        }
        
        # Ajout des probabilités si demandé
        if return_probabilities:
            probabilities = self.model.predict_proba(text_tfidf)[0]
            classes = self.label_encoder.classes_
            
            # Probabilité de la classe prédite
            result['confidence'] = float(probabilities[prediction_encoded])
            
            # Toutes les probabilités
            prob_dict = {}
            for i, class_name in enumerate(classes):
                prob_dict[class_name] = float(probabilities[i])
            
            # Tri par probabilité décroissante
            result['all_probabilities'] = dict(
                sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            )
        
        return result
    
    def predict_batch(self, examples: List[Dict], return_probabilities: bool = False) -> List[Dict]:
        """
        Fait des prédictions sur plusieurs exemples
        
        Args:
            examples (List[Dict]): Liste d'exemples avec les clés 'code_dtc', 'description', 'root_cause'
            return_probabilities (bool): Si True, retourne les probabilités
            
        Returns:
            List[Dict]: Liste des résultats de prédiction
        """
        if not self.is_loaded:
            self.load_model()
        
        results = []
        for example in examples:
            try:
                result = self.predict_single(
                    code_dtc=example.get('code_dtc', ''),
                    description=example.get('description', ''),
                    root_cause=example.get('root_cause', ''),
                    return_probabilities=return_probabilities
                )
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'input': example
                })
        
        return results
    
    def get_top_predictions(self, code_dtc: str, description: str, 
                           root_cause: str = "", top_n: int = 3) -> List[Dict]:
        """
        Retourne les top N prédictions avec leurs probabilités
        
        Args:
            code_dtc (str): Code DTC
            description (str): Description du problème
            root_cause (str): Description de la cause racine (optionnel)
            top_n (int): Nombre de prédictions à retourner
            
        Returns:
            List[Dict]: Liste des top prédictions
        """
        result = self.predict_single(code_dtc, description, root_cause, return_probabilities=True)
        
        if 'error' in result:
            return [result]
        
        top_predictions = []
        for pca, probability in list(result['all_probabilities'].items())[:top_n]:
            top_predictions.append({
                'pca': pca,
                'probability': probability,
                'confidence_level': self._get_confidence_level(probability)
            })
        
        return top_predictions
    
    def _get_confidence_level(self, probability: float) -> str:
        """
        Détermine le niveau de confiance basé sur la probabilité
        
        Args:
            probability (float): Probabilité de la prédiction
            
        Returns:
            str: Niveau de confiance
        """
        if probability >= 0.8:
            return "Très élevée"
        elif probability >= 0.6:
            return "Élevée"
        elif probability >= 0.4:
            return "Moyenne"
        elif probability >= 0.2:
            return "Faible"
        else:
            return "Très faible"
    
    def explain_prediction(self, code_dtc: str, description: str, root_cause: str = "") -> Dict:
        """
        Fournit une explication détaillée de la prédiction
        
        Args:
            code_dtc (str): Code DTC
            description (str): Description du problème
            root_cause (str): Description de la cause racine (optionnel)
            
        Returns:
            Dict: Explication détaillée
        """
        result = self.predict_single(code_dtc, description, root_cause, return_probabilities=True)
        
        if 'error' in result:
            return result
        
        explanation = {
            'prediction_summary': {
                'predicted_pca': result['predicted_pca'],
                'confidence': result['confidence'],
                'confidence_level': self._get_confidence_level(result['confidence'])
            },
            'input_analysis': {
                'original_input': result['input'],
                'processed_text': result['processed_text'],
                'text_length': len(result['processed_text']),
                'word_count': len(result['processed_text'].split())
            },
            'alternative_solutions': []
        }
        
        # Ajout des solutions alternatives
        for pca, prob in list(result['all_probabilities'].items())[1:4]:  # Top 3 alternatives
            explanation['alternative_solutions'].append({
                'pca': pca,
                'probability': prob,
                'confidence_level': self._get_confidence_level(prob)
            })
        
        return explanation


def main():
    """Fonction principale pour tester les prédictions"""
    predictor = PCAPredictor()
    
    # Exemples de test
    test_examples = [
        {
            'code_dtc': 'P0300',
            'description': 'Engine misfiring randomly',
            'root_cause': 'Faulty spark plugs'
        },
        {
            'code_dtc': 'P0171',
            'description': 'System too lean',
            'root_cause': 'Vacuum leak in intake manifold'
        }
    ]
    
    try:
        print("=== TEST DU PRÉDICTEUR PCA ===")
        
        for i, example in enumerate(test_examples, 1):
            print(f"\n--- Exemple {i} ---")
            print(f"Code DTC: {example['code_dtc']}")
            print(f"Description: {example['description']}")
            print(f"Root Cause: {example['root_cause']}")
            
            # Prédiction simple
            result = predictor.predict_single(**example)
            
            if 'error' not in result:
                print(f"PCA prédite: {result['predicted_pca']}")
                print(f"Confiance: {result['confidence']:.3f}")
                
                # Top 3 prédictions
                top_preds = predictor.get_top_predictions(**example, top_n=3)
                print("\nTop 3 prédictions:")
                for j, pred in enumerate(top_preds, 1):
                    print(f"  {j}. {pred['pca']} (prob: {pred['probability']:.3f}, confiance: {pred['confidence_level']})")
            else:
                print(f"Erreur: {result['error']}")
    
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        print("Assurez-vous d'avoir d'abord entraîné le modèle avec train_model.py")


if __name__ == "__main__":
    main()
