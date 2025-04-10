import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
import faiss

class ModelManager:
    def __init__(self, model_path: str = None):
        # Use absolute path based on current file location
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(base_dir, "models", "aesthetic_classifier.pkl")
        else:
            # If path is relative, make it absolute based on current directory
            if not os.path.isabs(model_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.model_path = os.path.join(base_dir, model_path.lstrip("../"))
            else:
                self.model_path = model_path
        
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained classifier model from disk"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded successfully from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
    
    def predict(self, embedding: np.ndarray):
        """Predict aesthetic category and probabilities"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Reshape for single sample prediction if needed
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = self.model.predict(embedding)[0]
        probabilities = self.model.predict_proba(embedding)[0]
        
        # Create probabilities dictionary
        prob_dict = {
            cat: float(prob) 
            for cat, prob in zip(self.model.classes_, probabilities)
        }
        
        # Map brand prediction to aesthetic category
        aesthetic_mapping = {
            # Luxury and high-end brands
            'Balmain': 'Dark Luxury',
            'Alexander McQueen': 'Dark Luxury',
            'Givenchy': 'Dark Luxury',
            
            # Streetwear brands
            'Nike': 'Streetwear',
            'Supreme': 'Streetwear',
            'BAPE': 'Streetwear',
            
            # Default to the original prediction if no mapping exists
            'default': prediction
        }
        
        mapped_prediction = aesthetic_mapping.get(prediction, aesthetic_mapping['default'])
        
        return {
            "prediction": mapped_prediction,
            "confidence": float(max(probabilities)),
            "probabilities": prob_dict,
            "original_brand": prediction
        }
    
    def predict_proba(self, embedding: np.ndarray) -> Dict[str, float]:
        """
        Get prediction probabilities for all categories
        
        Args:
            embedding (np.ndarray): Image embedding vector
            
        Returns:
            Dict[str, float]: Dictionary mapping category names to probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Reshape for single sample prediction if needed
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Get prediction probabilities
        proba = self.model.predict_proba(embedding)[0]
        
        # Map to category names
        result = {cat: float(prob) for cat, prob in zip(self.model.classes_, proba)}
        
        return result


class SimilaritySearch:
    def __init__(self, index_path: str = None, mapping_path: str = None):
        # Use absolute path based on current file location
        if index_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.index_path = os.path.join(base_dir, "models", "faiss_index.bin")
        else:
            # If path is relative, make it absolute based on current directory
            if not os.path.isabs(index_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.index_path = os.path.join(base_dir, index_path.lstrip("../"))
            else:
                self.index_path = index_path
        
        if mapping_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.mapping_path = os.path.join(base_dir, "models", "index_mapping.pkl")
        else:
            # If path is relative, make it absolute based on current directory
            if not os.path.isabs(mapping_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.mapping_path = os.path.join(base_dir, mapping_path.lstrip("../"))
            else:
                self.mapping_path = mapping_path
        
        self.index = None
        self.mapping = None
        self.load_index()
    
    def load_index(self):
        """Load the FAISS index and mapping from disk"""
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            self.index = faiss.read_index(self.index_path)
            
            with open(self.mapping_path, 'rb') as f:
                self.mapping = pickle.load(f)
            print(f"FAISS index loaded successfully from {self.index_path}")
        else:
            print(f"FAISS index or mapping not found. Similarity search will not be available.")
    
    def find_similar(self, embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar images based on embedding
        
        Args:
            embedding (np.ndarray): Image embedding vector
            k (int): Number of similar items to return
            
        Returns:
            List[Dict[str, Any]]: List of similar items with metadata
        """
        if self.index is None or self.mapping is None:
            raise ValueError("FAISS index or mapping not loaded")
        
        # Reshape for search if needed
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Convert to float32 if needed
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        # Search for similar embeddings
        distances, indices = self.index.search(embedding, k)
        
        # Get the metadata for each similar item
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.mapping) and idx >= 0:
                item = self.mapping[idx].copy()
                item['distance'] = float(distances[0][i])
                results.append(item)
        
        return results