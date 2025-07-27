"""
Démonstration complète du système PCA + GIM Chatbot
Montre les deux fonctionnalités principales de l'application
Auteur: Assistant IA
Date: 2025-07-26
"""

import pandas as pd
from predict import PCAPredictor
from gim_chatbot import GIMChatbot, demo_mode_response
from config import Config
import json
import os


def demo_header():
    """Affiche l'en-tête de démonstration"""
    print("=" * 80)
    print("🚀 DÉMONSTRATION COMPLÈTE - PRÉDICTEUR PCA & ASSISTANT GIM")
    print("=" * 80)
    print("🔧 Système intégré combinant :")
    print("   1. 🎯 Prédiction de solutions techniques (PCA) par IA")
    print("   2. 🤖 Assistant intelligent pour la plateforme GIM")
    print("=" * 80)


def demo_pca_predictions():
    """Démonstration du prédicteur PCA"""
    print("\n🎯 PARTIE 1 : PRÉDICTEUR PCA - DIAGNOSTIC AUTOMOBILE")
    print("-" * 60)
    
    # Initialisation du prédicteur
    predictor = PCAPredictor()
    
    try:
        predictor.load_model()
        print("✅ Modèle PCA chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur chargement modèle PCA: {e}")
        return
    
    # Exemples de diagnostic automobile
    automotive_examples = [
        {
            "name": "Problème de catalyseur",
            "code_dtc": "P0420",
            "description": "Catalytic converter efficiency below threshold",
            "root_cause": "Catalytic converter degraded due to age"
        },
        {
            "name": "Ratés d'allumage",
            "code_dtc": "P0300",
            "description": "Engine misfiring randomly",
            "root_cause": "Faulty spark plugs"
        },
        {
            "name": "Système trop pauvre",
            "code_dtc": "P0171",
            "description": "System too lean",
            "root_cause": "Vacuum leak in intake manifold"
        }
    ]
    
    print(f"\n🔍 Test sur {len(automotive_examples)} cas de diagnostic :")
    
    for i, example in enumerate(automotive_examples, 1):
        print(f"\n📋 CAS {i}: {example['name']}")
        print(f"   Code DTC: {example['code_dtc']}")
        print(f"   Problème: {example['description']}")
        print(f"   Cause: {example['root_cause']}")
        
        # Prédiction
        result = predictor.predict_single(
            code_dtc=example['code_dtc'],
            description=example['description'],
            root_cause=example['root_cause'],
            return_probabilities=True
        )
        
        if 'error' not in result:
            print(f"   🔧 PCA recommandée: {result['predicted_pca']}")
            print(f"   📊 Confiance: {result['confidence']:.1%}")
            
            # Top 2 alternatives
            if result['all_probabilities']:
                alternatives = list(result['all_probabilities'].items())[1:3]
                print(f"   🔄 Alternatives:")
                for j, (pca, prob) in enumerate(alternatives, 1):
                    print(f"      {j}. {pca[:50]}... ({prob:.1%})")
        else:
            print(f"   ❌ Erreur: {result['error']}")


def demo_gim_chatbot():
    """Démonstration du chatbot GIM"""
    print("\n\n🤖 PARTIE 2 : ASSISTANT GIM - GLOBAL ISSUE MANAGEMENT")
    print("-" * 60)
    
    # Initialisation du chatbot
    chatbot = GIMChatbot()
    
    print(f"✅ Chatbot GIM initialisé")
    print(f"📡 API disponible: {chatbot.is_available}")
    print(f"🎭 Mode: {'Production' if chatbot.is_available else 'Démonstration'}")
    
    # Questions de test sur GIM
    gim_questions = [
        "C'est quoi une PCA dans GIM ?",
        "Que signifient les statuts colorés dans GIM ?",
        "Qui peut clôturer un incident GIM ?",
        "Comment passer d'une ICA à une PCA ?",
        "Qu'est-ce qu'un Clean Point ?"
    ]
    
    print(f"\n🔍 Test sur {len(gim_questions)} questions GIM :")
    
    for i, question in enumerate(gim_questions, 1):
        print(f"\n❓ QUESTION {i}: {question}")
        
        # Obtenir la réponse
        if chatbot.is_available:
            response = chatbot.get_response(question)
        else:
            response = demo_mode_response(question)
        
        # Afficher la réponse (tronquée)
        print(f"🤖 RÉPONSE: {response[:200]}...")
        if len(response) > 200:
            print("   [Réponse complète disponible dans l'interface Streamlit]")


def demo_integration_benefits():
    """Montre les avantages de l'intégration"""
    print("\n\n🎉 PARTIE 3 : AVANTAGES DE L'INTÉGRATION")
    print("-" * 60)
    
    print("🔗 **Synergie PCA + GIM :**")
    print("   ✅ Interface unique pour diagnostic ET gestion d'incidents")
    print("   ✅ Workflow complet : Diagnostic → PCA → Suivi GIM")
    print("   ✅ Formation intégrée aux outils Capgemini")
    print("   ✅ Réduction du temps de résolution des incidents")
    
    print("\n📊 **Métriques de Performance :**")
    
    # Statistiques du modèle PCA
    try:
        results_files = [f for f in os.listdir('models') if f.startswith('pipeline_results_')]
        if results_files:
            latest_results = sorted(results_files)[-1]
            with open(f'models/{latest_results}', 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            if 'evaluation' in results:
                accuracy = results['evaluation']['metrics']['accuracy']
                print(f"   🎯 Accuracy PCA: {accuracy:.1%}")
            
            if 'preprocessing' in results:
                total_samples = results['preprocessing']['statistics']['total_samples']
                total_classes = results['preprocessing']['statistics']['total_classes']
                print(f"   📈 Dataset: {total_samples:,} exemples, {total_classes} classes PCA")
    except:
        print("   📊 Métriques disponibles dans l'interface Streamlit")
    
    print(f"\n🤖 **Chatbot GIM :**")
    gim_chatbot = GIMChatbot()
    print(f"   💡 {len(gim_chatbot.get_suggested_questions())} questions prédéfinies")
    print(f"   🔧 Mode démo fonctionnel sans configuration")
    print(f"   📝 Historique de conversation intégré")
    print(f"   🎯 Réponses contextuelles sur tous les aspects GIM")


def demo_usage_scenarios():
    """Scénarios d'utilisation pratiques"""
    print("\n\n💼 PARTIE 4 : SCÉNARIOS D'UTILISATION")
    print("-" * 60)
    
    scenarios = [
        {
            "title": "Technicien Automobile",
            "description": "Diagnostic rapide + recherche de PCA + suivi incident GIM",
            "steps": [
                "1. Saisit code DTC P0420 dans le prédicteur",
                "2. Obtient PCA 'Replace catalytic converter'",
                "3. Demande au chatbot 'Comment créer un GIM ?'",
                "4. Suit le processus GIM pour traçabilité"
            ]
        },
        {
            "title": "Ingénieur Qualité",
            "description": "Analyse des tendances + formation équipe GIM",
            "steps": [
                "1. Analyse l'historique des prédictions PCA",
                "2. Identifie les pannes récurrentes",
                "3. Utilise le chatbot pour former l'équipe sur GIM",
                "4. Optimise les processus qualité"
            ]
        },
        {
            "title": "Manager Opérationnel",
            "description": "Supervision + reporting + formation",
            "steps": [
                "1. Supervise les diagnostics via l'interface",
                "2. Consulte les métriques de performance",
                "3. Forme les nouveaux sur GIM via le chatbot",
                "4. Optimise les workflows d'incident"
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n👤 SCÉNARIO {i}: {scenario['title']}")
        print(f"   📝 {scenario['description']}")
        print(f"   🔄 Workflow:")
        for step in scenario['steps']:
            print(f"      {step}")


def demo_next_steps():
    """Prochaines étapes suggérées"""
    print("\n\n🚀 PROCHAINES ÉTAPES SUGGÉRÉES")
    print("-" * 60)
    
    print("🎯 **Pour tester immédiatement :**")
    print("   1. streamlit run app.py")
    print("   2. Tester le prédicteur PCA avec vos codes DTC")
    print("   3. Ouvrir l'Assistant GIM et poser vos questions")
    print("   4. Explorer l'historique et les fonctionnalités")
    
    print("\n🔧 **Pour la production :**")
    print("   1. Configurer la clé API Generative Engine")
    print("   2. Intégrer avec vos systèmes existants")
    print("   3. Former les équipes sur les nouvelles fonctionnalités")
    print("   4. Collecter les retours utilisateurs")
    
    print("\n📈 **Pour l'amélioration continue :**")
    print("   1. Analyser les logs d'utilisation")
    print("   2. Enrichir le dataset avec de nouveaux cas")
    print("   3. Étendre le chatbot à d'autres plateformes")
    print("   4. Développer des intégrations API")


def main():
    """Fonction principale de démonstration"""
    demo_header()
    
    try:
        demo_pca_predictions()
        demo_gim_chatbot()
        demo_integration_benefits()
        demo_usage_scenarios()
        demo_next_steps()
        
        print("\n" + "=" * 80)
        print("✅ DÉMONSTRATION TERMINÉE AVEC SUCCÈS !")
        print("=" * 80)
        print("🎉 Le système PCA + GIM est prêt pour la production !")
        print("🌐 Lancez 'streamlit run app.py' pour l'interface complète")
        print("📖 Consultez GUIDE_CHATBOT_GIM.md pour plus de détails")
        
    except Exception as e:
        print(f"\n❌ Erreur durant la démonstration: {e}")
        print("💡 Assurez-vous que le modèle PCA est entraîné")


if __name__ == "__main__":
    main()
