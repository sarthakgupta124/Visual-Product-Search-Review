"""
Search Engine Module : This is Cross Modal Search Engine for Multimodal Product Search and Rating Prediction

This module implements the core search functionality for the multimodal product search system.
It handles:
- Image-based similarity search
- Text-based similarity search
- Rating prediction based on similar reviews
- Embedding management and caching
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from typing import Tuple, Dict, Optional
from src.logger import get_logger
from src.utils import get_custom_paths, get_training_variables
import faiss
from src.components.inference.inference_helper import (
    load_encoders,
    get_text_embedding,
    get_image_embedding
)
from transformers import BertTokenizer

logger = get_logger(__name__)


class SearchEngine:
    """
    Main search engine for multimodal product search and rating prediction.
    
    Attributes:
        image_encoder: ResNet50-based image encoder
        text_encoder: BERT-based text encoder
        df: DataFrame containing all reviews with metadata
        tokenizer: BERT tokenizer
        params: Training parameters
        paths: Custom paths configuration
        text_faiss_index: Pre-computed FAISS index for text
        image_faiss_index: Pre-computed FAISS index for images
    """
    
    def __init__(self):
        """Initialize the search engine by loading models, dataset, and pre-computed FAISS indexes."""
        try:
            logger.info("Initializing SearchEngine...")
            
            # Load configuration
            self.params = get_training_variables()
            self.paths = get_custom_paths()
            
            # Load models
            logger.info("Loading encoders...")
            self.text_encoder, self.image_encoder = load_encoders()
            logger.info("[OK] Encoders loaded successfully")
            
            # Load tokenizer
            logger.info("Loading BERT tokenizer...")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            logger.info("[OK] Tokenizer loaded successfully")
            
            # Load dataset
            logger.info("Loading dataset...")
            data_path = self.paths['ingestion_data_path']
            if os.path.exists(data_path):
                self.df = pd.read_csv(data_path)
            else:
                # Try alternative path
                data_path = self.paths['whole_data_path']
                self.df = pd.read_csv(data_path)
            
            # Clean data
            self.df = self.df.dropna(subset=['review_text', 'rating']).reset_index(drop=True)
            logger.info(f"[OK] Dataset loaded: {len(self.df)} reviews")
            
            # Load Pre-computed FAISS Indexes directly from artifacts
            logger.info("Loading pre-computed FAISS indexes...")
            text_index_path = "artifacts/faiss_text_index.bin"
            image_index_path = "artifacts/faiss_image_index.bin"
            
            if not os.path.exists(text_index_path) or not os.path.exists(image_index_path):
                raise FileNotFoundError("FAISS index files are missing from the artifacts/ directory. Please compute them locally and push via Git LFS.")
                
            self.text_faiss_index = faiss.read_index(text_index_path)
            self.image_faiss_index = faiss.read_index(image_index_path)
            logger.info("[OK] FAISS Indexes loaded successfully")
            
            logger.info("[OK] SearchEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Error initializing SearchEngine: {str(e)}")
            raise
    
    def _compute_review_embeddings(self) -> np.ndarray:
        """
        Compute embeddings for all reviews in the dataset.
        NOTE: This is retained as a utility function for local regeneration, 
        but is deliberately bypassed in production initialization.
        """
        try:
            batch_size = self.params['BATCH_SIZE']
            max_len = self.params['MAX_LEN']
            all_embeddings = []
            
            review_texts = self.df['review_text'].values
            num_batches = (len(review_texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(review_texts), batch_size):
                batch_texts = review_texts[i:i+batch_size].tolist()
                
                # Tokenize batch
                tokens = self.tokenizer(
                    batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="tf"
                )
                
                # Get embeddings
                embeddings = self.text_encoder(
                    [tokens["input_ids"], tokens["attention_mask"]],
                    training=False
                )
                
                all_embeddings.append(embeddings.numpy())
                
                batch_num = (i // batch_size) + 1
                logger.info(f"Processed batch {batch_num}/{num_batches}")
            
            return np.vstack(all_embeddings)
        
        except Exception as e:
            logger.error(f"Error computing review embeddings: {str(e)}")
            raise
    
    def search_by_image(
        self,
        image_path: str,
        k: int = 5
    ) -> Dict:
        """
        Search for similar reviews based on a product image.
        """
        try:
            logger.info(f"Searching by image: {image_path}")
            
            # 1. Get image embedding
            image_embedding = get_image_embedding(image_path, self.image_encoder)[0].astype("float32")
    
            # 2. Search against the TEXT database (Cross-Modal Search)
            scores, indices = self.text_faiss_index.search(image_embedding.reshape(1, -1), k)
            
            # 3. Return the matching rows
            top_results = self.df.iloc[indices[0]].copy()
            return {"top_reviews": top_results, "similarities": scores[0]}
        
        except Exception as e:
            logger.error(f"Error in image search: {str(e)}")
            raise
    
    def search_by_text(
        self,
        text: str,
        k: int = 5
    ) -> Dict:
        """
        Search for similar reviews based on text query.
        """
        try:
            logger.info(f"Searching for images by text: {text[:50]}...")
            
            # 1. Get Text Embedding
            text_embedding = get_text_embedding(text, self.text_encoder)[0].astype("float32")
            
            # 2. CROSS-MODAL FIX: Search against the IMAGE database
            scores, indices = self.image_faiss_index.search(text_embedding.reshape(1, -1), k)
            
            # 3. Return matching rows
            top_results = self.df.iloc[indices[0]].copy()
            
            return {"top_reviews": top_results, "similarities": scores[0]}
        
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            raise
    
    def compute_final_rating(
        self,
        top_reviews: pd.DataFrame,
        similarities: np.ndarray,
        min_sim: float = 0.7
    ) -> float:
        """
        Compute the final predicted rating based on similar reviews.
        """
        try:
            ratings = top_reviews['rating'].values.astype(float)
            
            # Filter reviews by minimum similarity
            mask = similarities >= min_sim
            
            if mask.sum() == 0:
                # Fallback: use simple average of all reviews
                logger.warning(
                    f"No reviews with similarity >= {min_sim}. "
                    f"Using simple average of all {len(ratings)} reviews."
                )
                avg_rating = float(np.mean(ratings))
                logger.info(f"Computed final rating (average fallback): {avg_rating:.2f}/5.0")
                return avg_rating
            
            # Use weighted average of filtered reviews
            filtered_ratings = ratings[mask]
            filtered_similarities = similarities[mask]
            
            # Weighted average using similarity scores as weights
            weights = filtered_similarities / np.sum(filtered_similarities)
            weighted_rating = float(np.sum(filtered_ratings * weights))
            
            logger.info(
                f"Computed final rating (weighted avg): {weighted_rating:.2f}/5.0 "
                f"(based on {mask.sum()} reviews with sim >= {min_sim})"
            )
            
            return weighted_rating
        
        except Exception as e:
            logger.error(f"Error computing final rating: {str(e)}")
            raise
    
    def batch_search(
        self,
        queries: list,
        query_type: str = 'text',
        k: int = 5
    ) -> list:
        """
        Perform batch search for multiple queries.
        """
        results = []
        
        for idx, query in enumerate(queries, 1):
            try:
                if query_type == 'text':
                    result = self.search_by_text(query, k=k)
                else:
                    result = self.search_by_image(query, k=k)
                
                results.append(result)
                logger.info(f"[OK] Completed batch query {idx}/{len(queries)}")
            
            except Exception as e:
                logger.error(f"Error processing query {idx}: {str(e)}")
                results.append(None)
        
        return results
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the search engine.
        """
        stats = {
            'num_reviews': len(self.df),
            'embedding_dimension': self.text_faiss_index.d,
            'avg_rating': float(self.df['rating'].mean()),
            'rating_std': float(self.df['rating'].std()),
            'rating_min': float(self.df['rating'].min()),
            'rating_max': float(self.df['rating'].max()),
        }
        return stats