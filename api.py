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
        self._load_model()
    
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.categories_path, 'rb') as f:
            self.categories = pickle.load(f)
    
    def predict(self, image_tensor):
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

# Entry point
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)