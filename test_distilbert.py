#!/usr/bin/env python3
"""
Test du modèle DistilBERT pour vérifier qu'il fonctionne correctement
"""

from predict import PCAPredictor
import time

def test_distilbert():
    """Test du modèle DistilBERT"""
    print("🧠 Test du modèle DistilBERT...")
    
    # Initialiser le prédicteur DistilBERT
    predictor = PCAPredictor(backend="distilbert")
    
    try:
        # Charger le modèle
        start_time = time.time()
        predictor.load_model()
        load_time = time.time() - start_time
        print(f"✅ Modèle chargé en {load_time:.2f} secondes")
        
        # Test de prédiction
        test_cases = [
            {
                "code_dtc": "P0420",
                "description": "Catalyst System Efficiency Below Threshold",
                "root_cause": "Catalytic converter degraded"
            },
            {
                "code_dtc": "P0171",
                "description": "System Too Lean Bank 1",
                "root_cause": "Air leak in intake system"
            },
            {
                "code_dtc": "P0300",
                "description": "Random/Multiple Cylinder Misfire Detected",
                "root_cause": "Worn spark plugs"
            }
        ]
        
        print("\n🎯 Tests de prédiction:")
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test {i} ---")
            print(f"Code DTC: {test_case['code_dtc']}")
            print(f"Description: {test_case['description']}")
            
            start_time = time.time()
            result = predictor.predict_single(
                test_case['code_dtc'],
                test_case['description'],
                test_case['root_cause'],
                return_probabilities=True
            )
            pred_time = time.time() - start_time
            
            if 'error' in result:
                print(f"❌ Erreur: {result['error']}")
            else:
                print(f"✅ PCA prédite: {result['predicted_pca']}")
                print(f"📊 Confiance: {result['confidence']:.3f}")
                print(f"⏱️ Temps de prédiction: {pred_time:.3f}s")
                
                # Afficher les top 3 prédictions
                if result['all_probabilities']:
                    top_3 = list(result['all_probabilities'].items())[:3]
                    print("🏆 Top 3 prédictions:")
                    for j, (pca, prob) in enumerate(top_3, 1):
                        print(f"  {j}. {pca}: {prob:.3f}")
        
        print("\n✅ Test DistilBERT terminé avec succès!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test DistilBERT: {e}")
        return False

def test_randomforest():
    """Test du modèle RandomForest pour comparaison"""
    print("\n🌲 Test du modèle RandomForest...")
    
    predictor = PCAPredictor(backend="randomforest")
    
    try:
        start_time = time.time()
        predictor.load_model()
        load_time = time.time() - start_time
        print(f"✅ Modèle chargé en {load_time:.2f} secondes")
        
        # Test simple
        start_time = time.time()
        result = predictor.predict_single(
            "P0420",
            "Catalyst System Efficiency Below Threshold",
            "Catalytic converter degraded"
        )
        pred_time = time.time() - start_time
        
        if 'error' in result:
            print(f"❌ Erreur: {result['error']}")
            return False
        else:
            print(f"✅ PCA prédite: {result['predicted_pca']}")
            print(f"📊 Confiance: {result['confidence']:.3f}")
            print(f"⏱️ Temps de prédiction: {pred_time:.3f}s")
            return True
            
    except Exception as e:
        print(f"❌ Erreur lors du test RandomForest: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Test des modèles PCA\n")
    
    # Test DistilBERT
    distilbert_ok = test_distilbert()
    
    # Test RandomForest
    rf_ok = test_randomforest()
    
    print(f"\n📊 Résumé des tests:")
    print(f"DistilBERT: {'✅ OK' if distilbert_ok else '❌ Échec'}")
    print(f"RandomForest: {'✅ OK' if rf_ok else '❌ Échec'}")
    
    if distilbert_ok:
        print("\n🎯 Recommandation: Votre modèle DistilBERT est prêt pour le déploiement!")
    elif rf_ok:
        print("\n🎯 Recommandation: Utilisez RandomForest comme fallback")
    else:
        print("\n❌ Aucun modèle ne fonctionne correctement")
