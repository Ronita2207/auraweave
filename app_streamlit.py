import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from matplotlib import colors
import plotly.express as px
import pandas as pd
import math
from io import BytesIO
import os

# Constants
API_ENDPOINT = "http://127.0.0.1:8000"
AESTHETIC_CATEGORIES = [
    "Minimalist", "Grunge", "Office Siren", "Bohemian", "Streetwear",
    "Cottagecore", "Y2K", "Dark Luxury", "Athleisure", "Club Siren",
    "Dark Romance", "Techwear", "Indie Sleaze", "Dark Academia", "Regencycore"
]

# Page config
st.set_page_config(
    page_title="AuraWeave Fashion Analyzer",
    page_icon="👗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .color-box {
        width: 50px;
        height: 50px;
        display: inline-block;
        margin: 5px;
        border: 1px solid #ddd;
    }
    .aesthetic-card {
        border: 1px solid #ddd;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def display_color_palette(colors_list):
    """Display color palette with actual colors"""
    if not colors_list:
        st.warning("No color palette available")
        return
        
    # Ensure we have a valid list of colors
    valid_colors = [color for color in colors_list if isinstance(color, str) and color.startswith('#')]
    if not valid_colors:
        st.warning("Invalid color palette format")
        return
    
    # Create fixed number of columns (e.g., 5)
    num_cols = min(5, len(valid_colors))
    cols = st.columns(num_cols)
    
    for idx, color in enumerate(valid_colors):
        with cols[idx % num_cols]:
            st.markdown(f"""
                <div style='text-align: center'>
                    <div class='color-box' style='background-color: {color}'></div>
                    <p>{color}</p>
                </div>
            """, unsafe_allow_html=True)

def find_similar_products(image_data):
    """Find similar products using API"""
    try:
        # First get the analysis result which includes embeddings
        analysis_response = requests.post(f"{API_ENDPOINT}/analyze", files={"file": image_data})
        if analysis_response.status_code != 200:
            st.error("Failed to analyze image")
            return None
            
        # Then use the existing model's prediction to find similar items
        similar_response = requests.get(
            f"{API_ENDPOINT}/similar",
            params={"brand": analysis_response.json().get("prediction", "")}
        )
        
        if similar_response.status_code == 200:
            return similar_response.json()
        else:
            st.warning("Could not find similar items")
            return None
            
    except Exception as e:
        st.error(f"Error finding similar products: {str(e)}")
        return None

# Update the display_similar_items function
def display_similar_items(similar_items):
    """Display grid of similar items with details"""
    if not similar_items:
        st.warning("No similar items found")
        return
    
    # Create three columns for the grid
    for i in range(0, len(similar_items), 3):
        cols = st.columns(3)
        # Get the next three items
        batch = similar_items[i:min(i + 3, len(similar_items))]
        
        for col, item in zip(cols, batch):
            with col:
                try:
                    # Display image from URL or path
                    if "image_url" in item:
                        st.image(item["image_url"], use_container_width=True)
                    elif "image_path" in item:
                        image = Image.open(item["image_path"])
                        st.image(image, use_container_width=True)
                    
                    # Display metadata
                    if "brand" in item:
                        st.markdown(f"**Brand:** {item['brand']}")
                    if "name" in item:
                        st.markdown(f"**Product:** {item['name']}")
                    if "price" in item:
                        st.markdown(f"**Price:** ${float(item['price']):.2f}")
                    if "colour" in item:
                        st.markdown(f"**Color:** {item['colour']}")
                    
                    # Display similarity score
                    if "similarity_score" in item:
                        score = float(item["similarity_score"])
                        st.progress(score)
                        st.caption(f"Similarity: {score:.1%}")
                        
                except Exception as e:
                    st.error(f"Error displaying item: {str(e)}")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Upload & Analyze", "Explore Aesthetics"])

if page == "Upload & Analyze":
    st.title("🎭 Fashion Style Analyzer")
    
    uploaded_file = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Fixed column widths
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Analyze Style & Find Similar"):
            with st.spinner("Analyzing image..."):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(f"{API_ENDPOINT}/analyze", files=files)
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        with col2:
                            st.markdown("### Analysis Results")
                            
                            # Ensure prediction exists and is valid
                            if "prediction" in results and results["prediction"]:
                                st.markdown(f"**Predicted Style:** {results['prediction']}")
                                
                                # Handle confidence score
                                if "confidence" in results and isinstance(results["confidence"], (int, float)):
                                    confidence = float(results["confidence"]) * 100
                                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                                    st.progress(min(1.0, float(results["confidence"])))
                                
                                # Display probabilities if available
                                if "probabilities" in results:
                                    st.markdown("### Style Probabilities")
                                    for style, prob in sorted(
                                        results["probabilities"].items(), 
                                        key=lambda x: x[1], 
                                        reverse=True
                                    )[:5]:
                                        st.markdown(f"- {style}: {prob*100:.1f}%")
                                
                                # Display color palette if available
                                if "color_palette" in results and results["color_palette"]:
                                    st.markdown("### Color Palette")
                                    display_color_palette(results["color_palette"])
                                
                                # Display style elements if available
                                if "style_elements" in results and results["style_elements"]:
                                    st.markdown("### Style Elements")
                                    for element in results["style_elements"]:
                                        st.markdown(f"- {element}")
                                
                                # Display recommended brands if available
                                if "recommended_brands" in results and results["recommended_brands"]:
                                    st.markdown("### Recommended Brands")
                                    for brand in results["recommended_brands"]:
                                        st.markdown(f"- {brand}")
                                
                                # Show similar items
                                st.markdown("### Similar Items")
                                similar_items = find_similar_products(uploaded_file.getvalue())
                                if similar_items and "similar_items" in similar_items:
                                    display_similar_items(similar_items["similar_items"])
                            else:
                                st.error("No valid prediction found in response")
                                st.json(results)  # Show raw response for debugging
                    else:
                        st.error(f"API Error: {response.status_code}")
                        try:
                            error_detail = response.json()
                            st.error(f"Error details: {error_detail}")
                        except:
                            st.error(f"Raw response: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the API server. Please ensure it's running.")
                    st.info(f"API server should be running at {API_ENDPOINT}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif page == "Explore Aesthetics":
    st.title("🎨 Explore Fashion Aesthetics")
    
    # Display aesthetic categories
    selected_aesthetic = st.selectbox("Choose an aesthetic to explore", AESTHETIC_CATEGORIES)
    
    try:
        # Get aesthetic details
        response = requests.get(f"{API_ENDPOINT}/aesthetics/{selected_aesthetic}")
        if response.status_code == 200:
            aesthetic_info = response.json()
            
            # Display aesthetic information
            st.markdown(f"### {selected_aesthetic} Style Guide")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Style elements
                if 'style_elements' in aesthetic_info:
                    st.markdown("#### Key Elements")
                    for element in aesthetic_info['style_elements']:
                        st.markdown(f"- {element}")
                else:
                    st.warning("Style elements not available for this aesthetic")
                    
                # Recommended brands
                if 'key_brands' in aesthetic_info:
                    st.markdown("#### Popular Brands")
                    for brand in aesthetic_info['key_brands']:
                        st.markdown(f"- {brand}")
                else:
                    st.warning("Brand recommendations not available")
                    
            with col2:
                # Color palette
                if 'color_palette' in aesthetic_info:
                    st.markdown("#### Signature Colors")
                    display_color_palette(aesthetic_info['color_palette'])
                else:
                    st.warning("Color palette not available")
                
                # Silhouettes
                if 'silhouettes' in aesthetic_info:
                    st.markdown("#### Typical Silhouettes")
                    for silhouette in aesthetic_info['silhouettes']:
                        st.markdown(f"- {silhouette}")
                else:
                    st.warning("Silhouette information not available")
                    
        else:
            st.error(f"Error: Could not fetch aesthetic information (Status code: {response.status_code})")
            
    except Exception as e:
        st.error(f"Error loading aesthetic information: {str(e)}")
        st.info("Please make sure the API server is running at " + API_ENDPOINT)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by AuraWeave")