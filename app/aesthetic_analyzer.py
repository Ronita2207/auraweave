from typing import Dict, List, Tuple
import numpy as np
from .aesthetic_definitions import AESTHETIC_CATEGORIES
from .utils import ModelManager

class AestheticAnalyzer:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.categories = AESTHETIC_CATEGORIES
    
    def analyze_image(self, embedding: np.ndarray) -> Dict:
        # Get prediction results
        prediction_results = self.model_manager.predict(embedding)
        prediction = prediction_results["prediction"]
        
        # Get detailed aesthetic information
        aesthetic_info = self.categories.get(prediction, {})
        
        return {
            "prediction": prediction,
            "confidence": prediction_results["confidence"],
            "style_elements": aesthetic_info.get("style_elements", []),
            "color_palette": aesthetic_info.get("color_palette", []),
            "recommended_brands": aesthetic_info.get("key_brands", []),
            "typical_silhouettes": aesthetic_info.get("silhouettes", []),
            "probabilities": prediction_results.get("probabilities", {})
        }