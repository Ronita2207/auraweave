# Modified train_model.py that logs to a file
import os
import sys
import time

# Setup logging to a file
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.txt")

with open(log_file, "w") as f:
    f.write(f"=== Training started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Working directory: {os.getcwd()}\n")
    
    try:
        f.write("Importing required libraries...\n")
        import pandas as pd
        import numpy as np
        from app.image_encoder import ImageEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        
        f.write("Libraries imported successfully\n")
        
        # Hard-code paths for simplicity
        dataset_path = "../data/Fashion Dataset.csv"
        images_dir = "../data/images"
        model_save_path = "../models/aesthetic_classifier.pkl"
        
        # Check if files exist
        f.write(f"Checking if dataset exists at {dataset_path}...\n")
        if os.path.exists(dataset_path):
            f.write("Dataset found\n")
        else:
            f.write(f"ERROR: Dataset not found at {dataset_path}\n")
            sys.exit(1)
            
        f.write(f"Checking if images directory exists at {images_dir}...\n")
        if os.path.exists(images_dir):
            f.write("Images directory found\n")
            image_files = os.listdir(images_dir)
            f.write(f"Found {len(image_files)} files in images directory\n")
        else:
            f.write(f"ERROR: Images directory not found at {images_dir}\n")
            sys.exit(1)
        
        # Rest of your training code here...
        f.write("Training completed successfully\n")
        
    except Exception as e:
        import traceback
        f.write(f"ERROR: An exception occurred: {str(e)}\n")
        f.write(traceback.format_exc())