import os
import sys

from src.logger import get_logger
logging = get_logger(__name__)
# import dill

from src.exception import CustomException
def get_training_variables():
    try:
        return {
            'BATCH_SIZE': 6,
            'EPOCHS': 10,
            'LEARNING_RATE': 1e-4,
            'MAX_LEN': 128,
            'EMBEDDING_DIM': 256
        }
    except Exception as e:
        raise CustomException(e, sys)
    
def get_custom_paths():
    try:
        return {
            'TEXT_ENCODER_PATH': os.path.join('artifacts', 'text_encoder'),
            'IMAGE_ENCODER_PATH': os.path.join('artifacts', 'image_encoder'),
            'MULTIMODAL_MODEL_PATH': os.path.join('artifacts', 'multimodal_model.h5'),
            'IMAGE_BASE_DIR': os.path.join('notebook', 'data', 'images') + os.sep,
            'CKPT_DIR': os.path.join('artifacts', 'checkpoints'),
            'ingestion_data_path': os.path.join('artifacts', 'ingested_data.csv'),
            'whole_data_path': os.path.join('notebook', 'data', 'amazon_multimodal_dataset_with_ratings.csv')
        }
    except Exception as e:
        raise CustomException(e, sys)