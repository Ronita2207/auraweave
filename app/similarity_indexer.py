import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any

class SimilarityIndexer:
    def __init__(self, embedding_dim: int = 512):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.mapping: List[Dict[str, Any]] = []
    
    def add_item(self, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Add an item to the index"""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Add to FAISS index
        self.index.add(embedding.astype(np.float32))
        
        # Add metadata to mapping
        self.mapping.append(metadata)
    
    def save(self, index_path: str, mapping_path: str):
        """Save the index and mapping to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save mapping
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.mapping, f)