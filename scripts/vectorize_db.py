import os
import numpy as np
import pandas as pd
import tensorflow as tf
from src.components.inference.inference_helper import load_encoders, get_image_embedding, get_text_embedding
from src.utils import get_custom_paths

paths = get_custom_paths()

def vectorize():
    text_enc, img_enc = load_encoders()
    df = pd.read_csv(paths['whole_data_path'])
    df = df.dropna(subset=["review_text", "rating"]).reset_index(drop=True)
    
    print(f"Vectorizing {len(df)} items (Images and Reviews)...")
    img_embs = []
    txt_embs = []
    
    for idx, row in df.iterrows():
        raw_path = row['image_path']
        filename = os.path.basename(raw_path) 
        
        # Now join correctly: notebook/data/images/ + filename.jpg
        full_img_path = os.path.join(paths['IMAGE_BASE_DIR'], filename)

        # 1. Vectorize Image
        try:
            img_embs.append(get_image_embedding(full_img_path, img_enc))
        except Exception as e:
            # If still not found, add a zero vector to maintain alignment with the CSV
            img_embs.append(np.zeros((1, 256)))

        # 2. Vectorize Review Text
        try:
            txt_embs.append(get_text_embedding(row['review_text'], text_enc))
        except Exception as e:
            txt_embs.append(np.zeros((1, 256)))
            
        if idx % 500 == 0: 
            print(f"Processed {idx} items...")

    # Save to artifacts
    np.save(os.path.join('artifacts', 'image_embeddings.npy'), np.vstack(img_embs))
    np.save(os.path.join('artifacts', 'review_embeddings.npy'), np.vstack(txt_embs))
    print("✅ All embeddings saved to artifacts/")

if __name__ == "__main__":
    vectorize()