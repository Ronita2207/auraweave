# AuraWeave - Fashion Aesthetic Analysis API

AuraWeave is a machine learning backend that analyzes fashion outfit photos and predicts their aesthetic category (e.g., soft girl, grunge, minimal, etc.). This project uses ResNet18 for image feature extraction and a trained classifier to predict aesthetic categories.

## Features

- 📸 Image embedding extraction using ResNet18
- 🎨 Dominant color extraction from fashion images
- 🏷️ Aesthetic style classification
- 🔍 Similar outfit search using FAISS (optional)
- 🚀 Fast API endpoints for image analysis

## Project Structure

```
auraweave/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── image_encoder.py     # ResNet18 image embedding extraction
│   ├── color_extractor.py   # Dominant color extraction
│   └── utils.py             # Utility functions
│
├── data/
│   ├── Fashion Dataset.csv  # Your CSV file goes here
│   └── images/              # Your images folder goes here
│
├── models/
│   └── __init__.py          # Directory to store trained models
│
├── scripts/
│   └── train_model.py       # Script to train the classifier
│
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8+ (recommended)
- Pip package manager
- VS Code (or any preferred editor)

### 2. Clone/Set Up Project

Create the project directory structure as shown above.

### 3. Set Up Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Data Preparation

- Place your `Fashion Dataset.csv` file in the `data/` directory
- Place all your JPG images in the `data/images/` directory

### 5. Train the Model

```bash
cd scripts
python train_model.py
```

This will:
- Extract embeddings from all images using ResNet18
- Train a LogisticRegression classifier
- Save the model to `models/aesthetic_classifier.pkl`
- Create a FAISS index for similarity search (optional)

### 6. Run the API

```bash
cd app
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /upload/` - Upload and analyze a fashion image
- `POST /similar/` - Find similar outfits (if FAISS index is available)
- `GET /aesthetics/` - Get list of all aesthetic categories

## Example Usage

### Analyzing an outfit image

```python
import requests

url = "http://localhost:8000/upload/"
files = {"file": open("example_outfit.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

Response:
```json
{
    "filename": "example_outfit.jpg",
    "aesthetic": "minimal",
    "probabilities": {
        "minimal": 0.75,
        "soft girl": 0.15,
        "grunge": 0.05,
        "... other categories": "..."
    },
    "dominant_colors": ["#000000", "#ffffff", "#f5f5dc"]
}
```

## Customization

- To modify the number of dominant colors extracted, update the `num_colors` parameter in `ColorExtractor` initialization in `main.py`.
- To use a different model for embeddings, modify the `ImageEncoder` class in `image_encoder.py`.
- To add additional endpoints or features, extend the FastAPI application in `main.py`.

## License

MIT