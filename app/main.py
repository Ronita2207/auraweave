from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import time
from typing import List, Dict, Any, Optional
import io
import faiss
import numpy as np
import pickle
import pandas as pd

# Use relative imports for modules in the same package
from .image_encoder import ImageEncoder
from .color_extractor import ColorExtractor
from .utils import ModelManager, SimilaritySearch
from .aesthetic_analyzer import AestheticAnalyzer
from .aesthetic_definitions import AESTHETIC_CATEGORIES

app = FastAPI(
    title="AuraWeave API",
    description="API for fashion aesthetic analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get absolute paths to model files
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "aesthetic_classifier.pkl")
faiss_index_path = os.path.join(base_dir, "models", "faiss_index.bin")
mapping_path = os.path.join(base_dir, "models", "index_mapping.pkl")

# Initialize components
print(f"Loading model from: {model_path}")
model_manager = ModelManager(model_path=model_path)
image_encoder = ImageEncoder()
color_extractor = ColorExtractor(num_colors=3)

# Initialize FAISS index and mapping
try:
    index = faiss.read_index(faiss_index_path)
    with open(mapping_path, 'rb') as f:
        index_mapping = pickle.load(f)
    similarity_enabled = True
    print("FAISS index loaded successfully")
except Exception as e:
    print(f"Warning: Could not load FAISS index: {e}")
    similarity_enabled = False
    index = None
    index_mapping = None

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"message": "Welcome to AuraWeave API", "status": "online"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "similarity_search": similarity_enabled,
        "model_path": model_path,
        "model_exists": os.path.exists(model_path)
    }

@app.post("/upload/")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded fashion image
    
    - **file**: JPG image file
    
    Returns:
        Predicted aesthetic category and dominant colors
    """
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported")
    
    try:
        # Read file contents
        image_bytes = await file.read()
        
        # Extract embedding
        embedding = image_encoder.encode_image_from_bytes(image_bytes)
        
        # Get prediction
        aesthetic = model_manager.predict(embedding)
        
        # Get prediction probabilities
        probabilities = model_manager.predict_proba(embedding)
        
        # Extract top 3 dominant colors
        dominant_colors = color_extractor.extract_colors_from_bytes(image_bytes)
        
        return {
            "filename": file.filename,
            "aesthetic": aesthetic,
            "probabilities": probabilities,
            "dominant_colors": dominant_colors
        }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing image: {e}")
        print(error_details)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
@app.post("/similar/")
async def find_similar_outfits(file: UploadFile = File(...), k: int = 5):
    """Find similar outfits based on an uploaded image"""
    if not similarity_enabled:
        raise HTTPException(status_code=503, detail="Similarity search not available")
        
    try:
        # Read and process image
        contents = await file.read()
        embedding = image_encoder.encode_image_from_bytes(contents)
        
        # Reshape embedding for FAISS
        embedding = embedding.reshape(1, -1).astype('float32')
        
        # Search for similar items
        distances, indices = index.search(embedding, k)
        
        # Get metadata for similar items
        similar_items = []
        for i, idx in enumerate(indices[0]):
            if idx < len(index_mapping):
                item = index_mapping[idx].copy()
                item['distance'] = float(distances[0][i])
                item['similarity_score'] = 1.0 / (1.0 + item['distance'])
                similar_items.append(item)
        
        return {
            "similar_items": similar_items,
            "count": len(similar_items)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar")
async def get_similar_by_brand(brand: str):
    """Get similar items based on brand/style"""
    try:
        # Load the dataset
        dataset_path = os.path.join(os.path.dirname(base_dir), "data", "fashion_dataset.csv")
        df = pd.read_csv(dataset_path)
        
        # Filter by brand and get random samples
        similar_items = df[df['brand'].str.contains(brand, case=False, na=False)].sample(n=min(6, len(df)))
        
        # Prepare response
        items = []
        for _, item in similar_items.iterrows():
            items.append({
                "brand": item.get('brand', ''),
                "name": item.get('name', ''),
                "price": float(item.get('price', 0)),
                "colour": item.get('colour', ''),
                "image_url": item.get('img', ''),
                "similarity_score": 0.8  # Mock score since we're using brand matching
            })
        
        return {
            "similar_items": items,
            "count": len(items)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/aesthetics/")
async def get_aesthetics():
    """
    Get a list of all available aesthetic categories
    
    Returns:
        List of aesthetic categories
    """
    try:
        # Get the list of classes from the model
        categories = list(model_manager.model.classes_)
        return {"aesthetics": categories}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving aesthetics: {str(e)}")

@app.post("/analyze/")
async def analyze_fashion(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        embedding = image_encoder.encode_image_from_bytes(image_bytes)
        
        analyzer = AestheticAnalyzer(model_manager)
        analysis = analyzer.analyze_image(embedding)
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/aesthetics/{category}")
async def get_aesthetic_details(category: str):
    if category not in AESTHETIC_CATEGORIES:
        raise HTTPException(status_code=404, detail="Aesthetic category not found")
    
    return AESTHETIC_CATEGORIES[category]

@app.get("/aesthetics/")
async def list_aesthetics():
    return {
        "categories": list(AESTHETIC_CATEGORIES.keys()),
        "total_count": len(AESTHETIC_CATEGORIES)
    }

if __name__ == "__main__":
    # Run the app directly with uvicorn when the script is executed
    # Use the correct module path when running directly
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)