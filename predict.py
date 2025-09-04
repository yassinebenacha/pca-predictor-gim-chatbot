"""
Module de prédiction pour le projet NLP de prédiction de solutions techniques (PCA)
Permet de faire des prédictions sur de nouveaux exemples avec support DistilBERT et RandomForest
Auteur: Assistant IA
Date: 2025-07-26
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from typing import Dict, List, Tuple, Union, Optional
from preprocessing import TextPreprocessor

# Import conditionnel pour transformers
try:
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        pipeline
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class PCAPredictor:
    """
    Classe pour faire des prédictions de PCA sur de nouveaux exemples
    Support pour RandomForest (local) et DistilBERT (local ou Hugging Face)
    """

    def __init__(self, backend: str = "randomforest", model_dir: str = 'models',
                 checkpoint_dir: str = 'distilbert_pca_model', hf_repo: Optional[str] = None):
        """
        Initialise le prédicteur

        Args:
            backend (str): Type de modèle ('randomforest' ou 'distilbert')
            model_dir (str): Répertoire contenant les modèles RandomForest
            checkpoint_dir (str): Répertoire contenant le modèle DistilBERT local
            hf_repo (str): Repository Hugging Face pour DistilBERT (optionnel)
        """
        self.backend = backend.lower()
        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir
        self.hf_repo = hf_repo

        # Modèles RandomForest
        self.model = None
        self.vectorizer = None
        self.label_encoder = None

        # Modèles DistilBERT
        self.tokenizer = None
        self.distilbert_model = None
        self.label_mapping = None

        self.preprocessor = TextPreprocessor()
        self.is_loaded = False
    
    def load_model(self) -> None:
        """
        Charge le modèle selon le backend choisi
        """
        if self.backend == "randomforest":
            self._load_randomforest_model()
        elif self.backend == "distilbert":
            self._load_distilbert_model()
        else:
            raise ValueError(f"Backend non supporté: {self.backend}")

    def _load_randomforest_model(self) -> None:
        """Charge le modèle RandomForest"""
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

    def _load_distilbert_model(self) -> None:
        """Charge le modèle DistilBERT"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers non disponible. Installez avec: pip install transformers torch")

        try:
            # Essayer d'abord le modèle local
            if os.path.exists(self.checkpoint_dir):
                print(f"Chargement du modèle DistilBERT local depuis {self.checkpoint_dir}")
                self.tokenizer = DistilBertTokenizer.from_pretrained(self.checkpoint_dir)
                self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(self.checkpoint_dir)

                # Charger le mapping des labels
                label_mapping_path = os.path.join(self.checkpoint_dir, 'label_mapping.json')
                if os.path.exists(label_mapping_path):
                    with open(label_mapping_path, 'r', encoding='utf-8') as f:
                        self.label_mapping = json.load(f)
                else:
                    raise FileNotFoundError(f"Fichier label_mapping.json manquant dans {self.checkpoint_dir}")

            # Sinon essayer Hugging Face Hub
            elif self.hf_repo:
                print(f"Chargement du modèle DistilBERT depuis Hugging Face: {self.hf_repo}")
                self.tokenizer = DistilBertTokenizer.from_pretrained(self.hf_repo)
                self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(self.hf_repo)

                # Essayer de récupérer le label mapping depuis le repo
                try:
                    import requests
                    response = requests.get(f"https://huggingface.co/{self.hf_repo}/raw/main/label_mapping.json")
                    if response.status_code == 200:
                        self.label_mapping = response.json()
                    else:
                        raise Exception("Label mapping non trouvé sur HF Hub")
                except:
                    # Fallback avec mapping par défaut
                    print("⚠️ Utilisation du mapping de labels par défaut")
                    self.label_mapping = self._get_default_label_mapping()

            else:
                raise FileNotFoundError(f"Modèle DistilBERT non trouvé dans {self.checkpoint_dir} et aucun repo HF spécifié")

            self.distilbert_model.eval()
            self.is_loaded = True
            print(f"Modèle DistilBERT chargé avec succès")
            print(f"Classes disponibles: {len(self.label_mapping)} classes")

        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle DistilBERT: {e}")

    def _get_default_label_mapping(self) -> Dict:
        """Retourne un mapping de labels par défaut"""
        return {
            "0": "Replace Catalytic Converter",
            "1": "Check Air Filter and MAF Sensor",
            "2": "Replace Spark Plugs",
            "3": "Check EVAP System",
            "4": "Replace Thermostat"
        }

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

        if self.backend == "randomforest":
            return self._predict_single_randomforest(code_dtc, description, root_cause, return_probabilities)
        elif self.backend == "distilbert":
            return self._predict_single_distilbert(code_dtc, description, root_cause, return_probabilities)
        else:
            return {'error': f'Backend non supporté: {self.backend}'}

    def _predict_single_randomforest(self, code_dtc: str, description: str,
                                   root_cause: str = "", return_probabilities: bool = True) -> Dict:
        """Prédiction avec RandomForest"""
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

    def _predict_single_distilbert(self, code_dtc: str, description: str,
                                 root_cause: str = "", return_probabilities: bool = True) -> Dict:
        """Prédiction avec DistilBERT"""
        # Préprocessing de l'entrée
        processed_text = self.preprocess_input(code_dtc, description, root_cause)

        if not processed_text.strip():
            return {
                'error': 'Texte vide après préprocessing',
                'processed_text': processed_text
            }

        try:
            # Tokenisation
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            # Prédiction
            with torch.no_grad():
                outputs = self.distilbert_model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class_id = torch.argmax(logits, dim=-1).item()

            # Récupération du label prédit
            predicted_pca = self.label_mapping.get(str(predicted_class_id), f"Classe_{predicted_class_id}")

            result = {
                'input': {
                    'code_dtc': code_dtc,
                    'description': description,
                    'root_cause': root_cause
                },
                'processed_text': processed_text,
                'predicted_pca': predicted_pca,
                'confidence': float(probabilities[0][predicted_class_id]),
                'all_probabilities': None
            }

            # Ajout des probabilités si demandé
            if return_probabilities:
                prob_dict = {}
                for i, prob in enumerate(probabilities[0]):
                    label = self.label_mapping.get(str(i), f"Classe_{i}")
                    prob_dict[label] = float(prob)

                # Tri par probabilité décroissante
                result['all_probabilities'] = dict(
                    sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                )

            return result

        except Exception as e:
            return {
                'error': f'Erreur lors de la prédiction DistilBERT: {str(e)}',
                'processed_text': processed_text
            }

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
