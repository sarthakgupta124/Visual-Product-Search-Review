import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from src.components.inference.search_engine import SearchEngine
from src.utils import get_custom_paths, get_training_variables
from src.logger import get_logger
from src.components.explain.llm_explainer import explain_rating_cached

# Initialize logger
logger = get_logger(__name__)

# Page Configuration
st.set_page_config(
    page_title="Multimodal Product Search & Rating",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-card {
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        background-color: #f8f9fa;
    }
    .rating-badge {
        display: inline-block;
        background-color: #ffc107;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        color: black;
        margin: 0.5rem 0;
    }
    .similarity-score {
        color: #667eea;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def load_search_engine():
    """Load and cache the search engine"""
    try:
        logger.info("Loading SearchEngine...")
        search_engine = SearchEngine()
        logger.info("SearchEngine loaded successfully")
        return search_engine
    except Exception as e:
        logger.error(f"Error loading SearchEngine: {str(e)}")
        st.error(f"Failed to load SearchEngine: {str(e)}")
        return None

# Load search engine
search_engine = load_search_engine()

if search_engine is None:
    st.error("❌ Failed to initialize the search engine. Please check the logs.")
    st.stop()

# Main UI
st.markdown('<div class="main-title">🔍 Multimodal Product Search & Review Rating System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find products by image or text and get AI-powered ratings</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    search_type = st.radio(
        "Select Search Type:",
        ["Image Search", "Text Search"],
        help="Choose whether to search by product image or review text"
    )
    
    top_k = st.slider(
        "Number of Results (K):",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of similar reviews to retrieve"
    )
    
    min_similarity = st.slider(
        "Minimum Similarity Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Only consider reviews with similarity above this threshold for rating calculation"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    params = get_training_variables()
    st.metric("Embedding Dimension", params['EMBEDDING_DIM'])
    st.metric("Max Text Length", params['MAX_LEN'])

# Main Content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("📤 Input")
    
    if search_type == "Image Search":
        st.subheader("Upload Product Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a product image to search for similar reviews"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Uploaded Image")
            
            if st.button("🔍 Search by Image", use_container_width=True):
                with st.spinner("Processing image and searching..."):
                    try:
                        # Save temporary image
                        temp_path = "temp_image.jpg"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Get results
                        results = search_engine.search_by_image(
                            image_path=temp_path,
                            k=top_k
                        )
                        
                        # Calculate final rating
                        top_reviews = results['top_reviews']
                        similarities = results['similarities']
                        final_rating = search_engine.compute_final_rating(
                            top_reviews=top_reviews,
                            similarities=similarities,
                            min_sim=min_similarity
                        )

                        # AI explanation
                        st.info("Generating AI explanation with Groq...")
                        explanation = explain_rating_cached(final_rating, top_reviews, similarities)

                        st.session_state.search_results = {
                            'results': results,
                            'final_rating': final_rating,
                            'explanation': explanation,
                            'search_type': 'image'
                        }
                        
                        # Clean up
                        os.remove(temp_path)
                        
                        # Force Streamlit to refresh the page to show the results
                        st.experimental_rerun()
                        
                    except Exception as e:
                        logger.error(f"Error during image search: {str(e)}")
                        st.error(f"❌ Error during search: {str(e)}")
    
    else:  # Text Search
        st.subheader("Enter Review Text")
        user_text = st.text_area(
            "Type or paste a product review:",
            height=150,
            placeholder="e.g., Great product, very durable and fast shipping. Highly recommend!"
        )
        
        if st.button("🔍 Search by Text", use_container_width=True):
            if user_text.strip():
                with st.spinner("Processing text and searching..."):
                    try:
                        # Get results
                        results = search_engine.search_by_text(
                            text=user_text,
                            k=top_k
                        )
                        
                        # Calculate final rating
                        top_reviews = results['top_reviews']
                        similarities = results['similarities']
                        final_rating = search_engine.compute_final_rating(
                            top_reviews=top_reviews,
                            similarities=similarities,
                            min_sim=min_similarity
                        )
                        
                        st.info("Generating AI explanation with Groq...")
                        explanation = explain_rating_cached(final_rating, top_reviews, similarities)
                        
                        st.session_state.search_results = {
                            'results': results,
                            'final_rating': final_rating,
                            'explanation': explanation,
                            'search_type': 'text',
                            'query': user_text
                        }
                        
                        # Force Streamlit to refresh the page to show the results
                        st.experimental_rerun()
                        
                    except Exception as e:
                        logger.error(f"Error during text search: {str(e)}")
                        st.error(f"❌ Error during search: {str(e)}")
            else:
                st.warning("⚠️ Please enter some text to search")

with col2:
    st.header("📊 Results")
    
    if 'search_results' in st.session_state:
        results_data = st.session_state.search_results
        final_rating = results_data['final_rating']
        results = results_data['results']
        top_reviews = results['top_reviews']
        
        # Display Final Predicted Rating
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; color: #667eea;">🎯 Predicted Rating</h2>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align: center; margin: 2rem 0;">' 
            f'<span style="font-size: 4em; font-weight: bold; color: #ffc107;">⭐ {final_rating:.2f}</span>' 
            f'<br><span style="font-size: 1.5em; color: #666;">/5.0</span></div>',
            unsafe_allow_html=True
        )
        
        # Display metadata
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.metric("Reviews Analyzed", len(top_reviews))
        with col_meta2:
            st.metric("Rating Method", "Weighted Average")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display the AI Explanation
        st.markdown("### 🤖 Why this rating?")
        st.info(results_data['explanation'])
        
        # Display search query if text search
        if results_data['search_type'] == 'text':
            st.markdown("---")
            st.markdown("**Your Query:**")
            st.caption(results_data['query'])
    else:
        st.info("👈 Use the input panel to search for products by image or text")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>🚀 Powered by ResNet50 + BERT embeddings with contrastive learning</p>
    <p>Built with TensorFlow, Transformers, and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)