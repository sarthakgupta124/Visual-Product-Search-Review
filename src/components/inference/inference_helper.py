import tensorflow as tf
import numpy as np
import os
from transformers import BertTokenizer
from src.utils import get_training_variables, get_custom_paths

# Load configurations
params = get_training_variables()
paths = get_custom_paths()
MAX_LEN = params['MAX_LEN']

# Initialize Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_encoders():
    """Loads the saved SavedModel directories for both encoders."""
    text_encoder = tf.keras.models.load_model(paths['TEXT_ENCODER_PATH'])
    image_encoder = tf.keras.models.load_model(paths['IMAGE_ENCODER_PATH'])
    return text_encoder, image_encoder

def get_text_embedding(text, encoder):
    """Converts a string query into a normalized vector."""
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )
    embedding = encoder([tokens["input_ids"], tokens["attention_mask"]], training=False)
    return embedding.numpy()

def get_image_embedding(image_path, encoder):
    """Processes an image file and converts it into a normalized vector."""
    # Ensure the path uses the correct separators for Windows
    clean_path = os.path.normpath(image_path)
    
    img = tf.io.read_file(clean_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    
    embedding = encoder(img, training=False)
    return embedding.numpy()