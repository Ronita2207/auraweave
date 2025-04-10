import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

class EnhancedFashionEncoder(nn.Module):
    def __init__(self, num_aesthetics: int = 15):
        super().__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        
        # Replace classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_aesthetics)
        )
        
        self.backbone.fc = self.classifier

class ImageEncoder:
    def __init__(self):
        # Load pre-trained ResNet18 model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the final fully connected layer to get embeddings
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def encode_image(self, image_path):
        """
        Encode an image into a feature vector using ResNet18
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Open and transform the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Generate embedding
        with torch.no_grad():
            features = self.model(image_tensor)
        
        # Convert to numpy array and flatten
        embedding = features.squeeze().numpy().flatten()
        
        return embedding
    
    def encode_image_from_bytes(self, image_bytes):
        """
        Encode an image from bytes into a feature vector
        
        Args:
            image_bytes (bytes): Image data in bytes
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Open the image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Generate embedding
        with torch.no_grad():
            features = self.model(image_tensor)
        
        # Convert to numpy array and flatten
        embedding = features.squeeze().numpy().flatten()
        
        return embedding
