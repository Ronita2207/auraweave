import uvicorn
import os

if __name__ == "__main__":
    # Print some debug information
    print("=== Starting AuraWeave API ===")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "aesthetic_classifier.pkl")
    print(f"Working directory: {os.getcwd()}")
    print(f"Base directory: {base_dir}")
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    
    # Run the FastAPI app from the app package
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)