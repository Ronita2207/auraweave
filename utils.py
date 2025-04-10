import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from collections import Counter
import glob

class ImageProcessor:
    def __init__(self):
        print("Initializing ResNet model...")
        # Initialize ResNet model for embedding extraction
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("ResNet model initialized successfully")
    
    def extract_embedding(self, image_bytes):
        """Extract embedding from image bytes"""
        try:
            # Open the image from bytes
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            # Generate embedding
            with torch.no_grad():
                features = self.model(image_tensor)
            
            # Convert to numpy array and flatten
            embedding = features.squeeze().numpy().flatten()
            
            return embedding
        except Exception as e:
            print(f"Error in extract_embedding: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_colors(self, image_bytes, num_colors=3):
        """Extract dominant colors from image bytes"""
        try:
            # Open the image from bytes
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Resize image to speed up processing
            img = img.resize((150, 150))
            
            # Convert image to numpy array
            img_array = np.array(img)
            
            # Reshape the array to be a list of RGB values
            pixels = img_array.reshape(-1, 3)
            
            # Count similar colors
            counts = Counter()
            for pixel in pixels:
                # Quantize colors a bit to reduce the number of distinct colors
                quantized = tuple(map(lambda x: round(x / 25) * 25, pixel))
                counts[quantized] += 1
            
            # Get the most common colors
            dominant_colors = [self._rgb_to_hex(color) for color, _ in counts.most_common(num_colors)]
            
            return dominant_colors
        except Exception as e:
            print(f"Error in extract_colors: {e}")
            import traceback
            traceback.print_exc()
            # Return fallback colors in case of error
            return ["#000000", "#FFFFFF", "#808080"][:num_colors]
    
    def _rgb_to_hex(self, rgb):
        """Convert RGB tuple to hex color code"""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


class ModelManager:
    def __init__(self):
        print("Initializing ModelManager...")
        # Set paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, "models", "aesthetic_classifier.pkl")
        self.categories_path = os.path.join(self.base_dir, "models", "categories.pkl")
        
        print(f"Model path: {self.model_path}")
        print(f"Categories path: {self.categories_path}")
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model file not found at {self.model_path}")
            # Look for any pickle files in the models directory
            model_dir = os.path.join(self.base_dir, "models")
            if os.path.exists(model_dir):
                model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
                if model_files:
                    print(f"Found these model files instead: {model_files}")
                    # Use the first one
                    self.model_path = model_files[0]
                    print(f"Using {self.model_path} instead")
                else:
                    raise FileNotFoundError(f"Model not found at {self.model_path} and no alternative models found")
            else:
                print(f"Models directory not found at {model_dir}")
                raise FileNotFoundError(f"Models directory not found at {model_dir}")
        
        # Load model and categories
        self.model = self._load_model()
        self.categories = self._load_categories()
        print("ModelManager initialized successfully")
    
    def _load_model(self):
        """Load the trained classifier model"""
        print(f"Loading model from {self.model_path}...")
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully: {type(model)}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_categories(self):
        """Load the aesthetic categories"""
        if os.path.exists(self.categories_path):
            try:
                with open(self.categories_path, 'rb') as f:
                    categories = pickle.load(f)
                print(f"Loaded {len(categories)} categories: {categories[:5]}...")
                return categories
            except Exception as e:
                print(f"Error loading categories: {e}")
                # If we can't load categories, try to get them from the model
                if hasattr(self.model, 'classes_'):
                    print(f"Using model classes instead: {self.model.classes_}")
                    return list(self.model.classes_)
                return []
        else:
            print(f"Categories file not found at {self.categories_path}")
            # If categories file doesn't exist, try to get them from the model
            if hasattr(self.model, 'classes_'):
                print(f"Using model classes instead: {self.model.classes_}")
                return list(self.model.classes_)
            return []
    
    def predict(self, embedding):
        """Predict aesthetic category from embedding"""
        try:
            # Reshape for single sample prediction if needed
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Get prediction
            prediction = self.model.predict(embedding)[0]
            
            # Get probabilities
            probabilities = self.model.predict_proba(embedding)[0]
            
            # Map probabilities to categories
            probs_dict = {cat: float(prob) for cat, prob in zip(self.model.classes_, probabilities)}
            
            return prediction, probs_dict
        except Exception as e:
            print(f"Error in predict: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_categories(self):
        """Get list of aesthetic categories"""
        if not self.categories and hasattr(self.model, 'classes_'):
            return list(self.model.classes_)
        return self.categories