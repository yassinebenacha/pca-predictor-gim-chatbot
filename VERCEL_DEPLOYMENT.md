# 🚀 Guide de Déploiement Vercel

## 📋 Options de Déploiement

### **Option 1: RandomForest uniquement (Rapide)**
✅ **Recommandé pour un déploiement immédiat**

1. **Déployez directement sur Vercel**
   - Connectez-vous à [vercel.com](https://vercel.com)
   - Importez le repository: `yassinebenacha/pca-predictor-gim-chatbot`
   - Déployez sans configuration supplémentaire

2. **Fonctionnalités disponibles**
   - ✅ Prédicteur PCA avec 5 classes de base
   - ✅ Assistant GIM intégré
   - ✅ Interface complète
   - ✅ Temps de réponse < 1 seconde

### **Option 2: DistilBERT via Hugging Face (Optimal)**
🎯 **Recommandé pour les meilleures performances**

#### **Étape 1: Upload du modèle sur Hugging Face**

```bash
# 1. Installer huggingface_hub
pip install huggingface_hub

# 2. Se connecter à Hugging Face
huggingface-cli login

# 3. Uploader le modèle
python upload_to_hf.py
```

#### **Étape 2: Configuration Vercel**

1. **Variables d'environnement sur Vercel**
   ```
   HF_REPO=yassinebenacha/pca-distilbert-automotive
   ```

2. **Déployer sur Vercel**
   - Le modèle DistilBERT sera automatiquement téléchargé depuis HF Hub
   - 178 classes PCA disponibles
   - Haute précision

## ⚙️ Configuration Vercel

### **Paramètres optimisés (vercel.json)**
```json
{
  "functions": {
    "app.py": {
      "maxDuration": 300,
      "memory": 3008
    }
  }
}
```

### **Variables d'environnement optionnelles**
```
HF_REPO=votre-username/votre-modele-distilbert
GENERATIVE_ENGINE_API_KEY=votre-cle-api-gim
OPENAI_API_KEY=votre-cle-openai
```

## 🔧 Résolution des Problèmes

### **"❌ Modèle non disponible"**
- ✅ **Solution**: Le modèle RandomForest se crée automatiquement
- ✅ **Vérification**: Attendez 30-60 secondes après le déploiement
- ✅ **Alternative**: Configurez HF_REPO pour DistilBERT

### **Timeout lors du chargement**
- ✅ **Solution**: Configuré pour 5 minutes (suffisant)
- ✅ **Optimisation**: Utilisez RandomForest pour plus de rapidité

### **Erreur de mémoire**
- ✅ **Solution**: Configuré pour 3GB (maximum Vercel)
- ✅ **Alternative**: RandomForest utilise moins de mémoire

## 📊 Comparaison des Options

| Aspect | RandomForest | DistilBERT + HF |
|--------|-------------|-----------------|
| **Classes PCA** | 5 | 178 |
| **Précision** | Basique | Élevée |
| **Vitesse** | < 1s | 2-5s |
| **Mémoire** | < 500MB | < 2GB |
| **Configuration** | Aucune | HF_REPO |
| **Déploiement** | Immédiat | +5 min setup |

## 🎯 Recommandations

### **Pour un test rapide**
1. Déployez avec RandomForest (Option 1)
2. Testez l'interface et les fonctionnalités
3. Partagez le lien avec votre équipe

### **Pour la production**
1. Uploadez votre modèle sur Hugging Face (Option 2)
2. Configurez HF_REPO sur Vercel
3. Redéployez pour avoir les 178 classes PCA

## 🔗 Liens Utiles

- **Vercel Dashboard**: https://vercel.com/dashboard
- **Hugging Face Hub**: https://huggingface.co/
- **Documentation Vercel**: https://vercel.com/docs
- **Repository GitHub**: https://github.com/yassinebenacha/pca-predictor-gim-chatbot
