"""
Script de validation finale pour l'intégration PCA + GIM Chatbot
Vérifie que toutes les fonctionnalités sont opérationnelles
Auteur: Assistant IA
Date: 2025-07-26
"""

import os
import sys
import importlib
from datetime import datetime


def check_file_exists(filepath, description):
    """Vérifie l'existence d'un fichier"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MANQUANT")
        return False


def check_module_import(module_name, description):
    """Vérifie l'import d'un module"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - ERREUR: {e}")
        return False


def validation_structure_projet():
    """Valide la structure du projet"""
    print("🏗️ VALIDATION DE LA STRUCTURE DU PROJET")
    print("-" * 50)
    
    files_to_check = [
        # Fichiers principaux
        ("app.py", "Interface Streamlit principale"),
        ("main.py", "Pipeline ML principal"),
        ("predict.py", "Module de prédiction PCA"),
        ("preprocessing.py", "Module de préprocessing"),
        ("train_model.py", "Module d'entraînement"),
        
        # Nouveaux fichiers GIM
        ("gim_chatbot.py", "Module chatbot GIM"),
        ("config.py", "Configuration centralisée"),
        ("test_gim_integration.py", "Tests d'intégration"),
        
        # Scripts de démonstration
        ("demo.py", "Démonstration PCA"),
        ("demo_complet.py", "Démonstration complète"),
        ("augment_dataset.py", "Augmentation dataset"),
        
        # Configuration et documentation
        ("requirements.txt", "Dépendances Python"),
        (".env.example", "Template configuration"),
        ("README.md", "Documentation principale"),
        ("GUIDE_CHATBOT_GIM.md", "Guide chatbot GIM"),
        ("INTEGRATION_COMPLETE.md", "Résumé intégration"),
        
        # Données et modèles
        ("data/gim_diagnostic_dataset.csv", "Dataset original"),
        ("data/gim_diagnostic_dataset_augmented.csv", "Dataset augmenté"),
        ("models/model.pkl", "Modèle ML entraîné"),
        ("models/vectorizer.pkl", "Vectoriseur TF-IDF"),
        ("models/label_encoder.pkl", "Encodeur de labels")
    ]
    
    success_count = 0
    for filepath, description in files_to_check:
        if check_file_exists(filepath, description):
            success_count += 1
    
    print(f"\n📊 Résultat: {success_count}/{len(files_to_check)} fichiers présents")
    return success_count == len(files_to_check)


def validation_imports():
    """Valide les imports des modules"""
    print("\n🐍 VALIDATION DES IMPORTS PYTHON")
    print("-" * 50)
    
    modules_to_check = [
        # Modules principaux
        ("preprocessing", "Module de préprocessing"),
        ("train_model", "Module d'entraînement"),
        ("predict", "Module de prédiction"),
        ("main", "Pipeline principal"),
        
        # Nouveaux modules
        ("gim_chatbot", "Chatbot GIM"),
        ("config", "Configuration"),
        
        # Dépendances externes
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("openai", "OpenAI"),
        ("dotenv", "Python-dotenv"),
        ("plotly", "Plotly")
    ]
    
    success_count = 0
    for module_name, description in modules_to_check:
        if check_module_import(module_name, description):
            success_count += 1
    
    print(f"\n📊 Résultat: {success_count}/{len(modules_to_check)} modules importables")
    return success_count == len(modules_to_check)


def validation_fonctionnalites():
    """Valide les fonctionnalités principales"""
    print("\n🔧 VALIDATION DES FONCTIONNALITÉS")
    print("-" * 50)
    
    tests = []
    
    # Test 1: Prédicteur PCA
    try:
        from predict import PCAPredictor
        predictor = PCAPredictor()
        predictor.load_model()
        
        # Test de prédiction
        result = predictor.predict_single(
            code_dtc="P0420",
            description="Test problem",
            root_cause="Test cause"
        )
        
        if 'predicted_pca' in result:
            print("✅ Prédicteur PCA: Fonctionnel")
            tests.append(True)
        else:
            print("❌ Prédicteur PCA: Erreur de prédiction")
            tests.append(False)
            
    except Exception as e:
        print(f"❌ Prédicteur PCA: Erreur - {e}")
        tests.append(False)
    
    # Test 2: Chatbot GIM
    try:
        from gim_chatbot import GIMChatbot, demo_mode_response
        chatbot = GIMChatbot()
        
        # Test des questions suggérées
        questions = chatbot.get_suggested_questions()
        if len(questions) > 0:
            print("✅ Chatbot GIM: Questions suggérées OK")
        else:
            print("❌ Chatbot GIM: Pas de questions suggérées")
        
        # Test de réponse démo
        response = demo_mode_response("Test question")
        if response and len(response) > 10:
            print("✅ Chatbot GIM: Mode démo fonctionnel")
            tests.append(True)
        else:
            print("❌ Chatbot GIM: Mode démo défaillant")
            tests.append(False)
            
    except Exception as e:
        print(f"❌ Chatbot GIM: Erreur - {e}")
        tests.append(False)
    
    # Test 3: Configuration
    try:
        from config import Config
        status = Config.get_api_status()
        if status:
            print(f"✅ Configuration: {status}")
            tests.append(True)
        else:
            print("❌ Configuration: Erreur de statut")
            tests.append(False)
            
    except Exception as e:
        print(f"❌ Configuration: Erreur - {e}")
        tests.append(False)
    
    success_count = sum(tests)
    print(f"\n📊 Résultat: {success_count}/{len(tests)} fonctionnalités validées")
    return success_count == len(tests)


def validation_donnees():
    """Valide les données et modèles"""
    print("\n📊 VALIDATION DES DONNÉES ET MODÈLES")
    print("-" * 50)
    
    try:
        import pandas as pd
        
        # Dataset original
        df_original = pd.read_csv('data/gim_diagnostic_dataset.csv')
        print(f"✅ Dataset original: {len(df_original)} lignes, {len(df_original.columns)} colonnes")
        
        # Dataset augmenté
        df_augmented = pd.read_csv('data/gim_diagnostic_dataset_augmented.csv')
        print(f"✅ Dataset augmenté: {len(df_augmented)} lignes, {len(df_augmented.columns)} colonnes")
        
        # Vérification de l'amélioration
        improvement = len(df_augmented) / len(df_original)
        print(f"✅ Amélioration dataset: {improvement:.1f}x plus de données")
        
        # Vérification des classes
        classes_original = df_original['PCA attendue'].nunique()
        classes_augmented = df_augmented['PCA attendue'].nunique()
        print(f"✅ Classes PCA: {classes_original} → {classes_augmented}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation données: {e}")
        return False


def validation_logs():
    """Valide la présence des logs"""
    print("\n📝 VALIDATION DES LOGS")
    print("-" * 50)
    
    if os.path.exists('output.log'):
        try:
            with open('output.log', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"✅ Fichier de logs: {len(lines)} lignes")
            
            # Vérifier les logs récents
            recent_logs = [line for line in lines if '2025-07-26' in line]
            print(f"✅ Logs récents: {len(recent_logs)} entrées aujourd'hui")
            
            return True
        except Exception as e:
            print(f"❌ Erreur lecture logs: {e}")
            return False
    else:
        print("⚠️ Fichier de logs non trouvé (normal si première utilisation)")
        return True


def validation_finale():
    """Validation finale complète"""
    print("🎯 VALIDATION FINALE - INTÉGRATION PCA + GIM CHATBOT")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 60)
    
    # Exécution des validations
    validations = [
        ("Structure du projet", validation_structure_projet),
        ("Imports Python", validation_imports),
        ("Fonctionnalités", validation_fonctionnalites),
        ("Données et modèles", validation_donnees),
        ("Logs", validation_logs)
    ]
    
    results = []
    for name, validation_func in validations:
        try:
            result = validation_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Erreur lors de {name}: {e}")
            results.append(False)
    
    # Résumé final
    success_count = sum(results)
    total_count = len(results)
    
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DE LA VALIDATION")
    print("=" * 60)
    
    for i, (name, result) in enumerate(zip([v[0] for v in validations], results)):
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\n🎯 SCORE GLOBAL: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("\n🎉 VALIDATION COMPLÈTE RÉUSSIE !")
        print("✅ Le système PCA + GIM Chatbot est prêt pour la production")
        print("🚀 Commandes pour démarrer :")
        print("   - streamlit run app.py")
        print("   - python demo_complet.py")
    else:
        print("\n⚠️ VALIDATION PARTIELLE")
        print("🔧 Vérifiez les erreurs ci-dessus avant la mise en production")
    
    return success_count == total_count


if __name__ == "__main__":
    validation_finale()
