#!/usr/bin/env python3
"""
Script pour uploader le modèle DistilBERT vers Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo
import json

def upload_model_to_hf():
    """Upload le modèle vers Hugging Face Hub"""
    
    # Configuration
    model_dir = "distilbert_pca_model"
    repo_name = "yassinebenacha/pca-distilbert-automotive"  # Changez selon votre username
    
    print(f"🚀 Upload du modèle vers Hugging Face Hub: {repo_name}")
    
    # Vérifier que le modèle existe
    if not os.path.exists(model_dir):
        print(f"❌ Dossier {model_dir} non trouvé")
        return False
    
    try:
        # Initialiser l'API HF
        api = HfApi()
        
        # Créer le repository (si il n'existe pas)
        try:
            create_repo(repo_name, exist_ok=True)
            print(f"✅ Repository {repo_name} créé/vérifié")
        except Exception as e:
            print(f"⚠️ Repository existe déjà ou erreur: {e}")
        
        # Fichiers à uploader
        files_to_upload = [
            "config.json",
            "model.safetensors", 
            "tokenizer_config.json",
            "vocab.txt",
            "label_mapping.json",
            "special_tokens_map.json"
        ]
        
        # Upload des fichiers
        for file_name in files_to_upload:
            file_path = os.path.join(model_dir, file_name)
            if os.path.exists(file_path):
                print(f"📤 Upload de {file_name}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_name,
                    repo_id=repo_name,
                    repo_type="model"
                )
                print(f"✅ {file_name} uploadé")
            else:
                print(f"⚠️ {file_name} non trouvé")
        
        # Créer un README
        readme_content = f"""---
language: en
tags:
- automotive
- diagnostic
- pca
- distilbert
license: mit
---

# PCA Predictor DistilBERT Model

Ce modèle DistilBERT a été fine-tuné pour prédire les solutions techniques (PCA) dans le diagnostic automobile.

## Utilisation

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("{repo_name}")
model = DistilBertForSequenceClassification.from_pretrained("{repo_name}")

# Exemple de prédiction
text = "P0420 Catalyst System Efficiency Below Threshold"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

## Classes

Ce modèle prédit parmi 178 classes PCA différentes pour le diagnostic automobile.

## Performance

- **Accuracy**: Optimisé pour le diagnostic automobile
- **Classes**: 178 solutions techniques (PCA)
- **Dataset**: Diagnostic automobile augmenté
"""
        
        # Upload du README
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model"
        )
        
        print(f"✅ Modèle uploadé avec succès sur: https://huggingface.co/{repo_name}")
        print(f"🔗 Utilisez ce repo dans votre app: {repo_name}")
        
        return repo_name
        
    except Exception as e:
        print(f"❌ Erreur lors de l'upload: {e}")
        print("💡 Assurez-vous d'être connecté avec: huggingface-cli login")
        return False

if __name__ == "__main__":
    print("📤 Upload du modèle DistilBERT vers Hugging Face Hub")
    print("⚠️ Assurez-vous d'être connecté avec: huggingface-cli login")
    
    repo_name = upload_model_to_hf()
    
    if repo_name:
        print(f"\n🎯 Prochaines étapes:")
        print(f"1. Vérifiez votre modèle sur: https://huggingface.co/{repo_name}")
        print(f"2. Utilisez ce repo dans votre app Vercel")
        print(f"3. Configurez la variable d'environnement HF_REPO={repo_name}")
    else:
        print("\n❌ Upload échoué. Vérifiez votre connexion Hugging Face.")
