import os
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import pickle
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from pydantic import BaseModel

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    # Create FastAPI app
    app = FastAPI(
        title="AuraWeave API",
        description="API for fashion aesthetic analysis",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

# Create the FastAPI application
app = create_app()

# Create ImageProcessor class
class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    async def process_image(self, file):
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        return self.transform(image).unsqueeze(0)

# Create ModelManager class
class ModelManager:
    def __init__(self):
        self.model_path = os.path.join('models', 'aesthetic_classifier.pkl')
        self.categories_path = os.path.join('models', 'categories.pkl')
        try:
            self._load_model()
        except FileNotFoundError:
            logger.warning("Model files not found. Using mock responses.")
            self.model = None
            self.categories = None
    
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.categories_path, 'rb') as f:
            self.categories = pickle.load(f)
    
    def predict(self, image_tensor):
        if self.model is None:
            # Return mock response if model isn't available
            return "Minimalist"
        
        features = image_tensor.numpy().reshape(1, -1)
        return self.model.predict(features)[0]

# Initialize components
image_processor = ImageProcessor()
model_manager = ModelManager()

@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "AuraWeave API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict aesthetic category for an uploaded image"""
    try:
        image_tensor = await image_processor.process_image(file)
        prediction = model_manager.predict(image_tensor)
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze image and return detailed results"""
    try:
        image_tensor = await image_processor.process_image(file)
        prediction = model_manager.predict(image_tensor)
        
        # Mock response with additional details
        result = {
            "prediction": prediction,
            "confidence": 0.85,
            "probabilities": {
                "Minimalist": 0.85,
                "Dark Academia": 0.08,
                "Streetwear": 0.05,
                "Y2K": 0.01,
                "Bohemian": 0.01
            },
            "color_palette": ["#000000", "#ffffff", "#cccccc", "#777777", "#333333"],
            "style_elements": ["Clean lines", "Monochromatic palette", "Simple silhouettes", "Functional design"],
            "recommended_brands": ["COS", "Arket", "Uniqlo", "Everlane", "Muji"]
        }
        return result
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/aesthetics/{aesthetic}")
async def get_aesthetic_details(aesthetic: str):
    """Get details about a specific aesthetic"""
    aesthetic_info = {
        "Minimalist": {
            "style_elements": ["Clean lines", "Monochromatic palette", "Simple silhouettes", "Functional design"],
            "key_brands": ["COS", "Arket", "Uniqlo", "Everlane", "Muji"],
            "color_palette": ["#000000", "#ffffff", "#cccccc", "#777777", "#333333"],
            "silhouettes": ["Straight", "Boxy", "Relaxed", "Tailored"]
        },
        # Add other aesthetics with mock data
    }
    
    if aesthetic in aesthetic_info:
        return aesthetic_info[aesthetic]
    else:
        # Return default mock data
        return {
            "style_elements": ["Signature elements for " + aesthetic],
            "key_brands": ["Brand 1", "Brand 2", "Brand 3"],
            "color_palette": ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff"],
            "silhouettes": ["Silhouette 1", "Silhouette 2"]
        }

@app.get("/similar")
async def get_similar_items(brand: str = ""):
    """Get similar fashion items"""
    # Mock response with similar items
    return {
        "similar_items": [
            {
                "image_url": "https://via.placeholder.com/300x400?text=Product+1",
                "brand": brand or "Fashion Brand",
                "name": "Stylish Product",
                "price": "59.99",
                "colour": "Black",
                "similarity_score": "0.92"
            },
            {
                "image_url": "https://via.placeholder.com/300x400?text=Product+2",
                "brand": brand or "Fashion Brand",
                "name": "Trendy Item",
                "price": "79.99",
                "colour": "White",
                "similarity_score": "0.87"
            },
            {
                "image_url": "https://via.placeholder.com/300x400?text=Product+3",
                "brand": brand or "Fashion Brand",
                "name": "Classic Piece",
                "price": "49.99",
                "colour": "Gray",
                "similarity_score": "0.81"
            }
        ]
    }

# Entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)