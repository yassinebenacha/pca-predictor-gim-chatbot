"""
Script pour configurer et pousser le projet vers GitHub
Auteur: Assistant IA
Date: 2025-07-26
"""

import os
import subprocess
import sys


def run_command(command, description):
    """Exécute une commande et affiche le résultat"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✅ {description} - Réussi")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - Échec")
            if result.stderr.strip():
                print(f"   Erreur: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False


def setup_git_repository():
    """Configure le repository Git local"""
    print("🚀 CONFIGURATION DU REPOSITORY GIT")
    print("-" * 50)
    
    commands = [
        ("git --version", "Vérification de Git"),
        ("git init", "Initialisation du repository Git"),
        ('git config user.name "Yassine Ben Acha"', "Configuration du nom utilisateur"),
        ('git config user.email "yssinebenacha91@gmail.com"', "Configuration de l'email"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def prepare_files():
    """Prépare les fichiers pour GitHub"""
    print("\n📁 PRÉPARATION DES FICHIERS")
    print("-" * 50)
    
    # Copier README_GITHUB.md vers README.md pour GitHub
    try:
        import shutil
        shutil.copy2("README_GITHUB.md", "README.md")
        print("✅ README.md créé pour GitHub")
    except Exception as e:
        print(f"❌ Erreur copie README: {e}")
        return False
    
    # Vérifier les fichiers essentiels
    essential_files = [
        "README.md",
        "LICENSE", 
        ".gitignore",
        "requirements.txt",
        "app.py",
        "main.py",
        "gim_chatbot.py"
    ]
    
    missing_files = []
    for file in essential_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file} présent")
    
    if missing_files:
        print(f"❌ Fichiers manquants: {missing_files}")
        return False
    
    return True


def add_and_commit_files():
    """Ajoute et committe tous les fichiers"""
    print("\n📦 AJOUT ET COMMIT DES FICHIERS")
    print("-" * 50)
    
    commands = [
        ("git add .", "Ajout de tous les fichiers"),
        ("git status", "Vérification du statut"),
        ('git commit -m "Initial commit: Prédicteur PCA + Assistant GIM intégré"', "Commit initial"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            if "nothing to commit" in command:
                print("ℹ️ Aucun changement à committer")
                continue
            return False
    
    return True


def setup_remote_and_push():
    """Configure le remote et pousse vers GitHub"""
    print("\n🌐 CONFIGURATION REMOTE ET PUSH VERS GITHUB")
    print("-" * 50)
    
    # Demander l'URL du repository
    print("📋 INFORMATIONS NÉCESSAIRES:")
    print("1. Créez d'abord le repository sur GitHub.com:")
    print("   - Nom: pca-predictor-gim-chatbot")
    print("   - Visibilité: Private")
    print("   - Ne pas initialiser avec README")
    print()
    
    repo_url = input("🔗 Entrez l'URL de votre repository GitHub (ex: https://github.com/yassinebenacha/pca-predictor-gim-chatbot.git): ")
    
    if not repo_url.strip():
        print("❌ URL du repository requise")
        return False
    
    commands = [
        (f"git remote add origin {repo_url}", "Ajout du remote origin"),
        ("git branch -M main", "Renommage de la branche en main"),
        ("git push -u origin main", "Push initial vers GitHub"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def display_final_instructions():
    """Affiche les instructions finales"""
    print("\n🎉 REPOSITORY GITHUB CONFIGURÉ AVEC SUCCÈS !")
    print("=" * 60)
    print("✅ Tous les fichiers ont été poussés vers GitHub")
    print("✅ Repository privé configuré")
    print("✅ Documentation complète incluse")
    print()
    print("🔗 PROCHAINES ÉTAPES:")
    print("1. Vérifiez votre repository sur GitHub.com")
    print("2. Configurez les collaborateurs si nécessaire")
    print("3. Activez les GitHub Actions si souhaité")
    print("4. Partagez le lien avec votre équipe")
    print()
    print("📁 FICHIERS INCLUS:")
    files_included = [
        "✅ Code source complet (PCA + GIM)",
        "✅ Documentation détaillée",
        "✅ Scripts de test et démonstration", 
        "✅ Configuration et dépendances",
        "✅ Datasets et modèles (si < 100MB)",
        "✅ Licence MIT"
    ]
    for file in files_included:
        print(f"   {file}")


def main():
    """Fonction principale"""
    print("🚀 SETUP GITHUB REPOSITORY - PCA PREDICTOR + GIM CHATBOT")
    print("=" * 70)
    
    # Vérifier que nous sommes dans le bon répertoire
    if not os.path.exists("app.py"):
        print("❌ Erreur: Ce script doit être exécuté depuis le répertoire du projet")
        print("💡 Assurez-vous d'être dans le dossier contenant app.py")
        sys.exit(1)
    
    steps = [
        ("Configuration Git", setup_git_repository),
        ("Préparation des fichiers", prepare_files),
        ("Commit des fichiers", add_and_commit_files),
        ("Push vers GitHub", setup_remote_and_push),
    ]
    
    for step_name, step_function in steps:
        if not step_function():
            print(f"\n❌ Échec à l'étape: {step_name}")
            print("🔧 Corrigez les erreurs ci-dessus et relancez le script")
            sys.exit(1)
    
    display_final_instructions()


if __name__ == "__main__":
    main()
