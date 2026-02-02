"""
Search Engine Module

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
        review_embeddings: Pre-computed embeddings for all reviews (numpy array)
        df: DataFrame containing all reviews with metadata
        tokenizer: BERT tokenizer
        params: Training parameters
        paths: Custom paths configuration
    """
    
    def __init__(self):
        """Initialize the search engine by loading models and embeddings."""
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
            
            # Load review embeddings
            logger.info("Loading pre-computed review embeddings...")
            embedding_path = self.paths['ingestion_data_path'].replace('.csv', '_embeddings.npy')
            
            if not os.path.exists(embedding_path):
                # Try alternative path
                embedding_path = os.path.join('artifacts', 'review_embeddings.npy')
            
            if os.path.exists(embedding_path):
                self.review_embeddings = np.load(embedding_path)
                logger.info(f"[OK] Loaded embeddings shape: {self.review_embeddings.shape}")
            else:
                logger.warning(f"Embedding file not found at {embedding_path}. Will compute on-the-fly.")
                self.review_embeddings = None
            
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
            
            # If embeddings not loaded, compute them
            if self.review_embeddings is None:
                logger.warning("Computing review embeddings (this may take a while)...")
                self.review_embeddings = self._compute_review_embeddings()
                logger.info(f"[OK] Computed embeddings shape: {self.review_embeddings.shape}")
            else:
                # Trim embeddings to match cleaned dataframe length
                if len(self.review_embeddings) > len(self.df):
                    logger.warning(f"Trimming embeddings from {len(self.review_embeddings)} to {len(self.df)} rows")
                    self.review_embeddings = self.review_embeddings[:len(self.df)]
                elif len(self.review_embeddings) < len(self.df):
                    logger.error(f"Embeddings ({len(self.review_embeddings)}) < DataFrame ({len(self.df)}). Recomputing...")
                    self.review_embeddings = self._compute_review_embeddings()
            
            logger.info("[OK] SearchEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Error initializing SearchEngine: {str(e)}")
            raise
    
    def _compute_review_embeddings(self) -> np.ndarray:
        """
        Compute embeddings for all reviews in the dataset.
        
        Returns:
            numpy array of shape (num_reviews, embedding_dim)
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
        
        Args:
            image_path: Path to the product image
            k: Number of top similar reviews to return
        
        Returns:
            Dictionary containing:
                - 'top_reviews': DataFrame of top k reviews
                - 'similarities': Array of similarity scores
                - 'image_embedding': The query image embedding
        """
        try:
            logger.info(f"Searching by image: {image_path}")
            
            # Get image embedding
            image_embedding = get_image_embedding(image_path, self.image_encoder)
            image_embedding = image_embedding[0]  # Remove batch dimension
            
            # Compute similarity scores with all reviews
            similarities = np.dot(self.review_embeddings, image_embedding)
            
            # Get top k indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            # Get top k reviews - ensure indices are valid
            valid_indices = top_k_indices[top_k_indices < len(self.df)]
            if len(valid_indices) < len(top_k_indices):
                logger.warning(f"Some indices out of bounds. Using {len(valid_indices)} valid indices.")
            
            top_reviews = self.df.iloc[valid_indices].copy()
            top_similarities = similarities[valid_indices]
            
            logger.info(f"[OK] Found {len(top_reviews)} similar reviews")
            
            return {
                'top_reviews': top_reviews,
                'similarities': top_similarities,
                'image_embedding': image_embedding,
                'scores': similarities
            }
        
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
        
        Args:
            text: Query text
            k: Number of top similar reviews to return
        
        Returns:
            Dictionary containing:
                - 'top_reviews': DataFrame of top k reviews
                - 'similarities': Array of similarity scores
                - 'text_embedding': The query text embedding
        """
        try:
            logger.info(f"Searching by text: {text[:50]}...")
            
            # Get text embedding
            text_embedding = get_text_embedding(text, self.text_encoder)
            text_embedding = text_embedding[0]  # Remove batch dimension
            
            # Compute similarity scores with all reviews
            similarities = np.dot(self.review_embeddings, text_embedding)
            
            # Get top k indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            # Get top k reviews - ensure indices are valid
            valid_indices = top_k_indices[top_k_indices < len(self.df)]
            if len(valid_indices) < len(top_k_indices):
                logger.warning(f"Some indices out of bounds. Using {len(valid_indices)} valid indices.")
            
            top_reviews = self.df.iloc[valid_indices].copy()
            top_similarities = similarities[valid_indices]
            
            logger.info(f"[OK] Found {len(top_reviews)} similar reviews")
            
            return {
                'top_reviews': top_reviews,
                'similarities': top_similarities,
                'text_embedding': text_embedding,
                'scores': similarities
            }
        
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
        
        Uses weighted average where:
        - Only reviews with similarity >= min_sim are considered
        - Weights are the similarity scores
        - If no reviews meet the threshold, falls back to simple average
        
        Args:
            top_reviews: DataFrame of top k similar reviews
            similarities: Array of similarity scores for top k reviews
            min_sim: Minimum similarity threshold (default 0.7)
        
        Returns:
            Predicted rating as a float (0.0 to 5.0)
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
        
        Args:
            queries: List of queries (text strings or image paths)
            query_type: 'text' or 'image'
            k: Number of results per query
        
        Returns:
            List of search results
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
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            'num_reviews': len(self.df),
            'embedding_dimension': self.review_embeddings.shape[1],
            'avg_rating': float(self.df['rating'].mean()),
            'rating_std': float(self.df['rating'].std()),
            'rating_min': float(self.df['rating'].min()),
            'rating_max': float(self.df['rating'].max()),
        }
        return stats
