import os
import numpy as np
import pandas as pd
import tensorflow as tf
from src.components.inference.inference_helper import load_encoders, get_image_embedding, get_text_embedding
from src.utils import get_custom_paths
from src.logger import get_logger

logger = get_logger(__name__)
paths = get_custom_paths()

def vectorize():
    """
    Vectorize review embeddings for fast inference.
    
    Uses the training dataset (ingested_data.csv) to generate embeddings.
    These embeddings must match the data used for training!
    
    Outputs:
    - image_embeddings.npy: Image embeddings (all reviews)
    - review_embeddings.npy: Text embeddings (all reviews)
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting embedding vectorization")
        logger.info("=" * 60)
        
        logger.info("Loading pre-trained encoders...")
        text_enc, img_enc = load_encoders()
        
        # Use the SAME data that was used for training!
        data_path = paths['ingestion_data_path']
        logger.info(f"Loading training dataset from: {data_path}")
        
        df = pd.read_csv(data_path)
        df = df.dropna(subset=["review_text", "rating"]).reset_index(drop=True)
        
        logger.info(f"Vectorizing {len(df)} items (Images and Reviews)...")
        img_embs = []
        txt_embs = []
        
        for idx, row in df.iterrows():
            raw_path = row['image_path']
            filename = os.path.basename(raw_path) 
            
            # Construct full image path
            full_img_path = os.path.join(paths['IMAGE_BASE_DIR'], filename)

            # 1. Vectorize Image
            try:
                img_embs.append(get_image_embedding(full_img_path, img_enc))
            except Exception as e:
                logger.warning(f"Failed to load image {full_img_path}: {str(e)}")
                # Add zero vector to maintain alignment with the CSV
                img_embs.append(np.zeros((1, 256)))

            # 2. Vectorize Review Text
            try:
                txt_embs.append(get_text_embedding(row['review_text'], text_enc))
            except Exception as e:
                logger.warning(f"Failed to embed text at row {idx}: {str(e)}")
                txt_embs.append(np.zeros((1, 256)))
                
            if (idx + 1) % 500 == 0: 
                logger.info(f"Progress: {idx + 1}/{len(df)} items processed...")

        # Stack and save embeddings
        img_embs_array = np.vstack(img_embs)
        txt_embs_array = np.vstack(txt_embs)
        
        artifacts_dir = os.path.join(paths['PROJECT_ROOT'], 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        
        img_path = os.path.join(artifacts_dir, 'image_embeddings.npy')
        txt_path = os.path.join(artifacts_dir, 'review_embeddings.npy')
        
        np.save(img_path, img_embs_array)
        np.save(txt_path, txt_embs_array)
        
        logger.info(f"✅ Saved image embeddings: {img_path} (shape: {img_embs_array.shape})")
        logger.info(f"✅ Saved text embeddings: {txt_path} (shape: {txt_embs_array.shape})")
        logger.info("=" * 60)
        logger.info("✅ Vectorization completed successfully")
        logger.info("=" * 60)
        
        return img_path, txt_path
        
    except Exception as e:
        logger.error(f"Error in vectorize: {str(e)}")
        raise

if __name__ == "__main__":
    vectorize()