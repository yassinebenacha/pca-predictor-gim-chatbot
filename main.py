"""
Pipeline principal pour le projet NLP de prédiction de solutions techniques (PCA)
Orchestre tout le processus : préprocessing, entraînement, évaluation et prédiction
Auteur: Assistant IA
Date: 2025-07-26
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any

from preprocessing import TextPreprocessor
from train_model import PCAPredictionModel
from predict import PCAPredictor


class PCAMLPipeline:
    """
    Pipeline principal pour le projet de prédiction de PCA
    """
    
    def __init__(self, data_path: str = 'data/gim_diagnostic_dataset.csv', 
                 model_dir: str = 'models'):
        """
        Initialise le pipeline
        
        Args:
            data_path (str): Chemin vers le fichier de données
            model_dir (str): Répertoire pour sauvegarder les modèles
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.preprocessor = TextPreprocessor()
        self.model = PCAPredictionModel()
        self.predictor = PCAPredictor(model_dir)
        
        # Création du répertoire de modèles
        os.makedirs(model_dir, exist_ok=True)
    
    def run_full_pipeline(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Exécute le pipeline complet : préprocessing + entraînement + évaluation
        
        Args:
            save_results (bool): Si True, sauvegarde les résultats
            
        Returns:
            Dict[str, Any]: Résultats complets du pipeline
        """
        print("=" * 60)
        print("🚀 DÉMARRAGE DU PIPELINE COMPLET PCA ML")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_path': self.data_path,
            'model_dir': self.model_dir,
            'preprocessing': {},
            'training': {},
            'evaluation': {},
            'feature_importance': {},
            'status': 'started'
        }
        
        try:
            # 1. Préprocessing des données
            print("\n📊 ÉTAPE 1: PRÉPROCESSING DES DONNÉES")
            print("-" * 40)
            
            df, X, y = self.preprocessor.load_and_preprocess_data(self.data_path)
            stats = self.preprocessor.get_data_statistics(df, X, y)
            
            results['preprocessing'] = {
                'status': 'success',
                'statistics': stats,
                'data_shape': (len(X), 1),
                'target_classes': y.nunique()
            }
            
            print(f"✅ Préprocessing terminé: {len(X)} exemples, {y.nunique()} classes")
            
            # 2. Préparation des données
            print("\n🔄 ÉTAPE 2: PRÉPARATION DES DONNÉES")
            print("-" * 40)
            
            X_train, X_test, y_train, y_test = self.model.prepare_data(X, y)
            
            # 3. Entraînement du modèle
            print("\n🧠 ÉTAPE 3: ENTRAÎNEMENT DU MODÈLE")
            print("-" * 40)
            
            train_metrics = self.model.train(X_train, y_train)
            results['training'] = {
                'status': 'success',
                'metrics': train_metrics
            }
            
            print(f"✅ Entraînement terminé avec accuracy CV: {train_metrics['cv_mean_accuracy']:.4f}")
            
            # 4. Évaluation du modèle
            print("\n📈 ÉTAPE 4: ÉVALUATION DU MODÈLE")
            print("-" * 40)
            
            eval_metrics = self.model.evaluate(X_test, y_test)
            results['evaluation'] = {
                'status': 'success',
                'metrics': eval_metrics
            }
            
            print(f"✅ Évaluation terminée avec accuracy: {eval_metrics['accuracy']:.4f}")
            
            # 5. Analyse des features importantes
            print("\n🔍 ÉTAPE 5: ANALYSE DES FEATURES")
            print("-" * 40)
            
            important_features = self.model.get_feature_importance(top_n=15)
            results['feature_importance'] = important_features
            
            print("Top 10 features les plus importantes:")
            for i, (feature, importance) in enumerate(list(important_features.items())[:10], 1):
                print(f"  {i:2d}. {feature:<20} : {importance:.4f}")
            
            # 6. Sauvegarde du modèle
            print("\n💾 ÉTAPE 6: SAUVEGARDE DU MODÈLE")
            print("-" * 40)
            
            self.model.save_model(self.model_dir)
            print("✅ Modèle sauvegardé avec succès")
            
            # 7. Test de prédiction
            print("\n🎯 ÉTAPE 7: TEST DE PRÉDICTION")
            print("-" * 40)
            
            test_result = self._test_prediction()
            results['prediction_test'] = test_result
            
            results['status'] = 'completed'
            
            # Sauvegarde des résultats
            if save_results:
                self._save_results(results)
            
            print("\n" + "=" * 60)
            print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
            print("=" * 60)
            print(f"📊 Accuracy finale: {eval_metrics['accuracy']:.4f}")
            print(f"📁 Modèle sauvegardé dans: {self.model_dir}")
            print(f"🔧 Prêt pour les prédictions!")
            
            return results
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"\n❌ ERREUR DANS LE PIPELINE: {e}")
            raise
    
    def _test_prediction(self) -> Dict[str, Any]:
        """
        Teste le système de prédiction avec un exemple
        
        Returns:
            Dict[str, Any]: Résultat du test
        """
        try:
            # Chargement du prédicteur
            self.predictor.load_model()
            
            # Exemple de test
            test_example = {
                'code_dtc': 'P0300',
                'description': 'Engine misfiring randomly',
                'root_cause': 'Faulty spark plugs'
            }
            
            # Prédiction
            result = self.predictor.predict_single(**test_example, return_probabilities=True)
            
            if 'error' not in result:
                print(f"✅ Test de prédiction réussi:")
                print(f"   Input: {test_example['code_dtc']} - {test_example['description']}")
                print(f"   PCA prédite: {result['predicted_pca']}")
                print(f"   Confiance: {result['confidence']:.3f}")
                
                return {
                    'status': 'success',
                    'test_input': test_example,
                    'prediction': result['predicted_pca'],
                    'confidence': result['confidence']
                }
            else:
                print(f"❌ Erreur lors du test de prédiction: {result['error']}")
                return {
                    'status': 'failed',
                    'error': result['error']
                }
                
        except Exception as e:
            print(f"❌ Erreur lors du test de prédiction: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Sauvegarde les résultats du pipeline
        
        Args:
            results (Dict[str, Any]): Résultats à sauvegarder
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.model_dir, f'pipeline_results_{timestamp}.json')
        
        # Conversion des numpy arrays en listes pour la sérialisation JSON
        def convert_numpy(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return obj
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=convert_numpy)
            print(f"📄 Résultats sauvegardés dans: {results_file}")
        except Exception as e:
            print(f"⚠️  Erreur lors de la sauvegarde des résultats: {e}")
    
    def predict_new_example(self, code_dtc: str, description: str, root_cause: str = "") -> Dict:
        """
        Fait une prédiction sur un nouvel exemple
        
        Args:
            code_dtc (str): Code DTC
            description (str): Description du problème
            root_cause (str): Description de la cause racine
            
        Returns:
            Dict: Résultat de la prédiction
        """
        try:
            if not self.predictor.is_loaded:
                self.predictor.load_model()
            
            return self.predictor.predict_single(code_dtc, description, root_cause)
        except Exception as e:
            return {'error': f"Erreur lors de la prédiction: {e}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne des informations sur le modèle
        
        Returns:
            Dict[str, Any]: Informations sur le modèle
        """
        model_files = ['model.pkl', 'vectorizer.pkl', 'label_encoder.pkl']
        model_status = {}
        
        for file in model_files:
            file_path = os.path.join(self.model_dir, file)
            model_status[file] = {
                'exists': os.path.exists(file_path),
                'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat() 
                          if os.path.exists(file_path) else None
            }
        
        return {
            'model_directory': self.model_dir,
            'data_path': self.data_path,
            'model_files': model_status,
            'is_trained': all(info['exists'] for info in model_status.values())
        }


def main():
    """Fonction principale avec interface en ligne de commande"""
    parser = argparse.ArgumentParser(description='Pipeline ML pour prédiction de PCA')
    parser.add_argument('--action', choices=['train', 'predict', 'info'], default='train',
                       help='Action à effectuer (default: train)')
    parser.add_argument('--data', default='data/gim_diagnostic_dataset.csv',
                       help='Chemin vers le fichier de données')
    parser.add_argument('--model-dir', default='models',
                       help='Répertoire des modèles')
    parser.add_argument('--code-dtc', help='Code DTC pour prédiction')
    parser.add_argument('--description', help='Description du problème')
    parser.add_argument('--root-cause', default='', help='Cause racine (optionnel)')
    
    args = parser.parse_args()
    
    # Initialisation du pipeline
    pipeline = PCAMLPipeline(args.data, args.model_dir)
    
    if args.action == 'train':
        # Entraînement complet
        results = pipeline.run_full_pipeline()
        
    elif args.action == 'predict':
        # Prédiction sur un nouvel exemple
        if not args.code_dtc or not args.description:
            print("❌ Pour la prédiction, --code-dtc et --description sont requis")
            sys.exit(1)
        
        result = pipeline.predict_new_example(args.code_dtc, args.description, args.root_cause)
        
        if 'error' not in result:
            print(f"🎯 PCA prédite: {result['predicted_pca']}")
            print(f"📊 Confiance: {result['confidence']:.3f}")
        else:
            print(f"❌ Erreur: {result['error']}")
    
    elif args.action == 'info':
        # Informations sur le modèle
        info = pipeline.get_model_info()
        print("📋 INFORMATIONS SUR LE MODÈLE")
        print("-" * 30)
        print(f"Répertoire: {info['model_directory']}")
        print(f"Données: {info['data_path']}")
        print(f"Modèle entraîné: {'✅ Oui' if info['is_trained'] else '❌ Non'}")
        
        for file, status in info['model_files'].items():
            status_icon = "✅" if status['exists'] else "❌"
            print(f"{status_icon} {file}: {status['size']} bytes")


if __name__ == "__main__":
    main()
