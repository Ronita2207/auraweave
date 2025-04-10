import os
import pandas as pd
import numpy as np
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time
import glob
import sys
from app.aesthetic_definitions import AESTHETIC_CATEGORIES
from app.image_encoder import EnhancedFashionEncoder
import torch.optim as optim
from app.similarity_indexer import SimilarityIndexer

# Modify these constants at the top of the file
TEST_MODE = False  # Change to False to use full dataset
MAX_SAMPLES = None  # Remove limit on samples

print("=== Training Started ===")
start_time = time.time()

# Debug file access
print("=== Debugging File Access ===")
print(f"Working directory: {os.getcwd()}")

# Use the specific data path you provided
data_dir = r"C:\Users\ronit\Downloads\data"
print(f"Data directory: {data_dir}")
print(f"Data directory exists: {os.path.exists(data_dir)}")

if os.path.exists(data_dir):
    print("Files in data directory:")
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        print(f"  - {file} (is file: {os.path.isfile(file_path)})")
        if file.lower().endswith('.csv'):
            print(f"    CSV file details: size={os.path.getsize(file_path)}, readable={os.access(file_path, os.R_OK)}")

# File paths - use the specific data path with nested Images directory
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(data_dir, "fashion_dataset.csv")  # Use the known good filename
images_dir = os.path.join(data_dir, "Images", "Images")  # Nested Images directory
model_dir = os.path.join(current_dir, "..", "models")  # Go up one level to models
model_path = os.path.join(model_dir, "aesthetic_classifier.pkl")

print(f"Dataset path: {dataset_path}")
print(f"Images directory: {images_dir}")
print(f"Model will be saved to: {model_path}")

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset not found at {dataset_path}")
    
    # Look for any CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if csv_files:
        print(f"Found these CSV files instead: {csv_files}")
        dataset_path = csv_files[0]
        print(f"Using {dataset_path} instead")
    else:
        print("No CSV files found in data directory")
        exit(1)

# Check if images directory exists
if not os.path.exists(images_dir):
    print(f"ERROR: Images directory not found at {images_dir}")
    
    # Try alternative paths
    alt_paths = [
        os.path.join(data_dir, "Images"),
        os.path.join(data_dir, "images"),
        os.path.join(data_dir, "Images", "images"),
        os.path.join(data_dir, "images", "Images")
    ]
    
    for path in alt_paths:
        if os.path.exists(path):
            images_dir = path
            print(f"Using alternative image path: {images_dir}")
            break
    
    if not os.path.exists(images_dir):
        print("Could not find a valid images directory. Please check the path.")
        exit(1)

# List contents of the images directory to verify
try:
    print(f"Contents of images directory ({images_dir}):")
    img_files = os.listdir(images_dir)
    print(f"Found {len(img_files)} files/directories")
    print(f"Sample items: {img_files[:5] if len(img_files) > 5 else img_files}")
except Exception as e:
    print(f"Error listing images directory: {e}")

# Create model directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Replace the dataset loading code
print("\n=== Loading Dataset ===")
print(f"Loading from: {dataset_path}")

df = pd.read_csv(dataset_path)
print(f"Total rows in CSV: {len(df)}")

if TEST_MODE:
    print(f"\n⚠️ TEST MODE ENABLED - Limiting to {MAX_SAMPLES} samples")
    df = df.head(MAX_SAMPLES).copy()  # .copy() to avoid SettingWithCopyWarning
    print(f"Dataset reduced to {len(df)} rows")

# Add this function to map URLs to local numerical filenames
def get_local_image_path(url_or_path, idx, images_dir):
    """Map URL or path to local numerical filename"""
    # First try direct numerical filename (e.g., 0.jpg, 1.jpg)
    for ext in ['.jpg', '.jpeg', '.png']:
        numerical_path = os.path.join(images_dir, f"{idx}{ext}")
        if os.path.exists(numerical_path):
            return numerical_path
    
    # Try with Unnamed:0 column if it exists
    if 'Unnamed: 0' in df.columns and idx < len(df):
        unnamed_idx = df.iloc[idx]['Unnamed: 0']
        for ext in ['.jpg', '.jpeg', '.png']:
            unnamed_path = os.path.join(images_dir, f"{unnamed_idx}{ext}")
            if os.path.exists(unnamed_path):
                return unnamed_path
            
    # If that fails and it's a URL, try extracting the ID from URL
    if isinstance(url_or_path, str) and url_or_path.startswith(('http://', 'https://')):
        try:
            # Extract ID from URL pattern like ".../images/17048614/..."
            parts = url_or_path.split('/')
            for part in parts:
                if part.isdigit():
                    for ext in ['.jpg', '.jpeg', '.png']:
                        id_path = os.path.join(images_dir, f"{part}{ext}")
                        if os.path.exists(id_path):
                            return id_path
        except Exception as e:
            if idx < 5:  # Only print for the first few
                print(f"Error extracting ID from URL: {e}")
    
    return None

# Check for 'img' column which likely contains image paths
if 'img' in df.columns:
    print(f"\nFound 'img' column which likely contains image paths")
    image_column = 'img'
    print(f"Sample from 'img' column: {df[image_column].iloc[:3].tolist()}")
else:
    # Check if image_name column exists
    if 'image_name' not in df.columns:
        print(f"\nWARNING: 'image_name' column not found in dataset.")
        
        # Try to find a suitable column
        possible_image_columns = [col for col in df.columns if any(s in col.lower() for s in ['image', 'file', 'filename', 'path', 'img'])]
        
        if possible_image_columns:
            image_column = possible_image_columns[0]
            print(f"Using '{image_column}' as the image filename column")
        else:
            # If no suitable column is found, use the first column
            image_column = df.columns[0]
            print(f"Using the first column '{image_column}' as the image filename column")
    else:
        image_column = 'image_name'

# Ensure image column values are strings
df[image_column] = df[image_column].astype(str)
print(f"\nConverted '{image_column}' column to strings")
print(f"Sample after conversion: {df[image_column].iloc[:3].tolist()}")

# Validate image paths before processing
print("\n=== Validating Images ===")
available_images = []
for idx, row in enumerate(df.itertuples()):
    if idx % 100 == 0:
        print(f"Checking images: {idx}/{len(df)}")
    
    image_path = get_local_image_path(getattr(row, image_column), idx, images_dir)
    if image_path and os.path.exists(image_path):
        available_images.append(idx)

print(f"\nFound {len(available_images)} valid images out of {len(df)} entries")

# Further limit dataset to only rows with available images
df = df.iloc[available_images].reset_index(drop=True)
print(f"Final dataset size: {len(df)} rows")

if len(df) == 0:
    print("Error: No valid images found in dataset!")
    sys.exit(1)

# For the label column, we need to find or create a suitable column
# Check if the dataset has a column that could serve as aesthetic categories
possible_label_columns = ['aesthetic', 'style', 'category', 'label', 'class', 'brand', 'colour']
found_label_column = False

for col in possible_label_columns:
    if col in df.columns:
        label_column = col
        found_label_column = True
        print(f"\nUsing '{label_column}' as the label column")
        break

if not found_label_column:
    # If we can't find a good label column, use 'brand' or 'colour' if available
    if 'brand' in df.columns:
        label_column = 'brand'
        print(f"\nUsing 'brand' as the label column")
    elif 'colour' in df.columns:
        label_column = 'colour'
        print(f"\nUsing 'colour' as the label column")
    else:
        # Last resort: use the last column
        label_column = df.columns[-1]
        print(f"\nNo obvious label column found. Using the last column '{label_column}' as the label column")

# Ensure label column values are strings and handle NaN values
df[label_column] = df[label_column].fillna('unknown').astype(str)
print(f"Unique values in '{label_column}' column: {df[label_column].nunique()}")
print(f"Sample categories: {df[label_column].value_counts().head(5).to_dict()}")

# Initialize image encoder
print("\nInitializing image encoder...")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()
print("ResNet18 model loaded")

# Extract embeddings
print("\nExtracting embeddings from images...")
embeddings = []
valid_indices = []
valid_image_paths = []

# Test a few mappings to verify the strategy
print("\nTesting image path mapping with the first few rows:")
for i in range(min(5, len(df))):
    img_col_value = df[image_column].iloc[i]
    print(f"Row {i}, Image column: {img_col_value}")
    matched_path = get_local_image_path(img_col_value, i, images_dir)
    if matched_path:
        print(f"✓ Successfully matched to: {matched_path}")
    else:
        print(f"✗ No match found")

# Process images
print("\nProcessing images for embeddings...")
for idx, row in enumerate(df.itertuples()):
    if idx % 100 == 0:
        print(f"Processing image {idx}/{len(df)}...")
    
    # Get image path using the new mapping function
    image_path = get_local_image_path(getattr(row, image_column), idx, images_dir)
    
    if not image_path:
        # If we still can't find the image, try by row index (Unnamed: 0)
        if hasattr(row, 'Unnamed: 0'):
            row_id = getattr(row, 'Unnamed: 0')
            image_path = os.path.join(images_dir, f"{row_id}.jpg")
            if not os.path.exists(image_path):
                image_path = None
    
    if not image_path:
        if idx < 5:  # Only show detailed messages for the first few
            print(f"Warning: Image not found for row {idx}, skipping.")
        continue
    
    try:
        # Open and transform the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Generate embedding
        with torch.no_grad():
            features = model(image_tensor)
        
        # Convert to numpy array and flatten
        embedding = features.squeeze().numpy().flatten()
        
        embeddings.append(embedding)
        valid_indices.append(idx)
        valid_image_paths.append(image_path)
        
        # Print a success message for the first few
        if idx < 5:
            print(f"Successfully processed: {image_path}")
            
    except Exception as e:
        if idx < 5:  # Only show detailed messages for the first few
            print(f"Error processing image {image_path}: {e}")

print(f"Successfully extracted embeddings for {len(embeddings)}/{len(df)} images")

if len(embeddings) == 0:
    print("No valid embeddings extracted. Aborting training.")
    exit(1)

# Create a new dataframe with only valid images
valid_df = df.iloc[valid_indices].reset_index(drop=True)

# Convert embeddings to numpy array
X = np.array(embeddings)
y = valid_df[label_column].values

print(f"Generated {len(embeddings)} embeddings with shape {X.shape}")
print(f"Unique categories: {np.unique(y)[:10] if len(np.unique(y)) > 10 else np.unique(y)}")

# Add minimum samples filter
min_samples_per_class = 2
class_counts = pd.Series(y).value_counts()
valid_classes = class_counts[class_counts >= min_samples_per_class].index
mask = pd.Series(y).isin(valid_classes)

# Filter X and y
X = X[mask]
y = y[mask]

print(f"\nAfter filtering classes with < {min_samples_per_class} samples:")
print(f"Remaining samples: {len(y)}")
print(f"Remaining classes: {len(np.unique(y))}")

# Now perform the train_test_split
print("\nSplitting data into training and test sets...")
if len(np.unique(y)) > 1:  # Only stratify if we have multiple classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples")

# Train a LogisticRegression classifier
print("Training classifier model...")
clf = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='multinomial')
clf.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model...")
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Save the model
print(f"Saving model to {model_path}...")
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)

# Save the unique aesthetic categories
categories = sorted(valid_df[label_column].unique().tolist())
categories_path = os.path.join(model_dir, "categories.pkl")
with open(categories_path, 'wb') as f:
    pickle.dump(categories, f)
print(f"Categories saved to {categories_path}")

# Create FAISS index for similarity search
print("\n=== Creating FAISS Index for Similarity Search ===")
try:
    import faiss
    
    # Convert embeddings to float32 (required by FAISS)
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Create a mapping from index to metadata
    index_mapping = []
    for idx, (row_idx, row) in enumerate(zip(valid_indices, valid_df.iterrows())):
        image_path = valid_image_paths[idx]
        index_mapping.append({
            'image_path': image_path,
            'aesthetic': row[1][label_column],  # Using row[1] to get the row data
            'index': idx,
            'original_index': row_idx,
            'filename': os.path.basename(image_path)
        })
    
    print(f"Created metadata mapping for {len(index_mapping)} images")
    
    # Create the FAISS index
    dimension = embeddings_array.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)    # Using L2 distance for similarity
    
    # Add embeddings to the index
    index.add(embeddings_array)
    print(f"Added {index.ntotal} vectors of dimension {dimension} to the index")
    
    # Save the index and mapping
    faiss_index_path = os.path.join(model_dir, "faiss_index.bin")
    mapping_path = os.path.join(model_dir, "index_mapping.pkl")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    
    faiss.write_index(index, faiss_index_path)
    with open(mapping_path, 'wb') as f:
        pickle.dump(index_mapping, f)
    
    print(f"FAISS index saved to {faiss_index_path}")
    print(f"Mapping saved to {mapping_path}")
    
    # Add similarity search info to the training summary
    with open(os.path.join(current_dir, "training_completed.txt"), "a") as f:
        f.write(f"FAISS index created with {index.ntotal} vectors\n")
        f.write(f"FAISS index saved to {faiss_index_path}\n")
        f.write(f"FAISS mapping saved to {mapping_path}\n")
    
except ImportError:
    print("Warning: faiss-cpu not available. Similarity search index not created.")
    print("To enable similarity search, install faiss-cpu: pip install faiss-cpu")
except Exception as e:
    print(f"Error creating FAISS index: {e}")
    import traceback
    traceback.print_exc()

# Build similarity index
print("\nBuilding similarity index...")
indexer = SimilarityIndexer(embedding_dim=512)

# Add embeddings to index with metadata
for idx, (embedding, row) in enumerate(zip(embeddings, valid_df.itertuples())):
    metadata = {
        "id": idx,
        "brand": getattr(row, "brand", "Unknown"),
        "name": getattr(row, "name", f"Item {idx}"),
        "price": float(getattr(row, "price", 0.0)),
        "image_path": valid_image_paths[idx],
        "colour": getattr(row, "colour", "Unknown"),
    }
    indexer.add_item(embedding, metadata)

# Save the index
index_path = os.path.join(model_dir, "faiss_index.bin")
mapping_path = os.path.join(model_dir, "index_mapping.pkl")
indexer.save(index_path, mapping_path)
print(f"Similarity index saved to {index_path}")

# Create a completion file
elapsed_time = time.time() - start_time
with open(os.path.join(current_dir, "training_completed.txt"), "w") as f:
    f.write("=== Training Summary ===\n")
    f.write(f"Training completed successfully in {elapsed_time:.2f} seconds\n")
    f.write(f"Processed {len(embeddings)} images out of {len(df)} total\n")
    f.write(f"Model saved to {model_path}\n")
    f.write(f"Model accuracy: {accuracy:.4f}\n")
    f.write(f"Categories: {categories[:10]}...\n" if len(categories) > 10 else f"Categories: {categories}\n")

print("=== Training Completed ===")
print(f"Took {elapsed_time:.2f} seconds")
print(f"Model saved to {model_path}")
print(f"Training summary written to {os.path.join(current_dir, 'training_completed.txt')}")

def train_model(train_loader, val_loader, num_epochs=30):
    model = EnhancedFashionEncoder(num_aesthetics=len(AESTHETIC_CATEGORIES))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )
    
    # ... training loop code ...