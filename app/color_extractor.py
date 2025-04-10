from PIL import Image
import numpy as np
from collections import Counter
import io

class ColorExtractor:
    def __init__(self, num_colors=3):
        self.num_colors = num_colors
    
    def rgb_to_hex(self, rgb):
        """Convert RGB tuple to hex color code"""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def extract_colors(self, image_path):
        """
        Extract dominant colors from an image file
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            list: List of hex color codes for the most dominant colors
        """
        # Open the image
        img = Image.open(image_path).convert('RGB')
        
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
        dominant_colors = [self.rgb_to_hex(color) for color, _ in counts.most_common(self.num_colors)]
        
        return dominant_colors
    
    def extract_colors_from_bytes(self, image_bytes):
        """
        Extract dominant colors from image bytes
        
        Args:
            image_bytes (bytes): Image data in bytes
            
        Returns:
            list: List of hex color codes for the most dominant colors
        """
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
        dominant_colors = [self.rgb_to_hex(color) for color, _ in counts.most_common(self.num_colors)]
        
        return dominant_colors