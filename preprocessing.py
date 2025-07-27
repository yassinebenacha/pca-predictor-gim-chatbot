"""
Module de préprocessing pour le projet NLP de prédiction de solutions techniques (PCA)
Auteur: Assistant IA
Date: 2025-07-26
"""

import pandas as pd
import re
import string
from typing import Tuple, List
import numpy as np


class TextPreprocessor:
    """
    Classe pour le préprocessing des données textuelles du diagnostic automobile
    """
    
    def __init__(self):
        """Initialise le préprocesseur avec les paramètres par défaut"""
        self.stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais', 
            'donc', 'car', 'ni', 'or', 'à', 'au', 'aux', 'avec', 'sans', 'sous', 
            'sur', 'dans', 'par', 'pour', 'en', 'vers', 'chez', 'contre', 'entre',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Nettoie un texte en supprimant la ponctuation, les caractères spéciaux
        et en normalisant la casse
        
        Args:
            text (str): Texte à nettoyer
            
        Returns:
            str: Texte nettoyé
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Conversion en string si ce n'est pas déjà le cas
        text = str(text)
        
        # Conversion en minuscules
        text = text.lower()
        
        # Suppression des caractères spéciaux et de la ponctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Suppression des chiffres isolés
        text = re.sub(r'\b\d+\b', '', text)
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Suppression des espaces en début et fin
        text = text.strip()
        
        return text
    
    def remove_stop_words(self, text: str) -> str:
        """
        Supprime les mots vides du texte
        
        Args:
            text (str): Texte à traiter
            
        Returns:
            str: Texte sans mots vides
        """
        if not text:
            return ''
        
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(filtered_words)
    
    def concatenate_text_columns(self, df: pd.DataFrame, 
                                columns: List[str] = None) -> pd.DataFrame:
        """
        Concatène les colonnes textuelles spécifiées en une seule colonne
        
        Args:
            df (pd.DataFrame): DataFrame contenant les données
            columns (List[str]): Liste des colonnes à concaténer
            
        Returns:
            pd.DataFrame: DataFrame avec la nouvelle colonne 'texte_concatene'
        """
        if columns is None:
            columns = ['Code DTC', 'Description du problème', 'Root Cause Description']
        
        df_copy = df.copy()
        
        # Nettoyage de chaque colonne
        for col in columns:
            if col in df_copy.columns:
                df_copy[col + '_clean'] = df_copy[col].apply(self.clean_text)
                df_copy[col + '_clean'] = df_copy[col + '_clean'].apply(self.remove_stop_words)
        
        # Concaténation des colonnes nettoyées
        clean_columns = [col + '_clean' for col in columns if col in df_copy.columns]
        df_copy['texte_concatene'] = df_copy[clean_columns].apply(
            lambda x: ' '.join(x.dropna().astype(str)), axis=1
        )
        
        return df_copy
    
    def load_and_preprocess_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Charge et préprocesse les données depuis un fichier CSV
        
        Args:
            file_path (str): Chemin vers le fichier CSV
            
        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series]: DataFrame complet, textes préprocessés, labels
        """
        # Chargement des données
        print(f"Chargement des données depuis {file_path}...")
        df = pd.read_csv(file_path)
        
        print(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
        print(f"Colonnes disponibles: {list(df.columns)}")
        
        # Vérification des colonnes nécessaires
        required_columns = ['Code DTC', 'Description du problème', 'Root Cause Description', 'PCA attendue']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Colonnes manquantes: {missing_columns}")
        
        # Suppression des lignes avec des valeurs manquantes dans les colonnes importantes
        df_clean = df.dropna(subset=['PCA attendue'])
        print(f"Après suppression des valeurs manquantes: {len(df_clean)} lignes")
        
        # Préprocessing du texte
        print("Préprocessing des colonnes textuelles...")
        df_processed = self.concatenate_text_columns(df_clean)
        
        # Extraction des features et labels
        X = df_processed['texte_concatene']
        y = df_processed['PCA attendue']
        
        # Suppression des lignes avec du texte vide
        mask = X.str.len() > 0
        X = X[mask]
        y = y[mask]
        
        print(f"Données finales: {len(X)} exemples")
        print(f"Nombre de classes uniques: {y.nunique()}")
        print(f"Classes: {sorted(y.unique())}")
        
        return df_processed, X, y
    
    def get_data_statistics(self, df: pd.DataFrame, X: pd.Series, y: pd.Series) -> dict:
        """
        Calcule des statistiques sur les données préprocessées
        
        Args:
            df (pd.DataFrame): DataFrame complet
            X (pd.Series): Textes préprocessés
            y (pd.Series): Labels
            
        Returns:
            dict: Dictionnaire contenant les statistiques
        """
        stats = {
            'total_samples': len(X),
            'total_classes': y.nunique(),
            'class_distribution': y.value_counts().to_dict(),
            'avg_text_length': X.str.len().mean(),
            'min_text_length': X.str.len().min(),
            'max_text_length': X.str.len().max(),
            'total_vocabulary': len(set(' '.join(X).split()))
        }
        
        return stats


def main():
    """Fonction principale pour tester le préprocessing"""
    preprocessor = TextPreprocessor()
    
    try:
        # Test du préprocessing
        df, X, y = preprocessor.load_and_preprocess_data('data/gim_diagnostic_dataset.csv')
        
        # Affichage des statistiques
        stats = preprocessor.get_data_statistics(df, X, y)
        print("\n=== STATISTIQUES DES DONNÉES ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Affichage de quelques exemples
        print("\n=== EXEMPLES DE DONNÉES PRÉPROCESSÉES ===")
        for i in range(min(3, len(X))):
            print(f"\nExemple {i+1}:")
            print(f"Texte: {X.iloc[i][:100]}...")
            print(f"PCA: {y.iloc[i]}")
        
    except Exception as e:
        print(f"Erreur lors du préprocessing: {e}")


if __name__ == "__main__":
    main()
