import tensorflow as tf
from src.exception import CustomException
import sys
from src.logger import get_logger
logging = get_logger(__name__)
from src.utils import get_training_variables
MAX_LEN = get_training_variables()['MAX_LEN']

from src.components.preprocessing.image_utils import load_image as load_image_tf
from src.components.preprocessing.text_utils import tokenize_py

from src.utils import get_custom_paths
CUSTOM_PATHS = get_custom_paths()
IMAGE_BASE_DIR = CUSTOM_PATHS['IMAGE_BASE_DIR']

def process_sample(image_path, review_text):
    try:
        # Image
        clean_image_path = tf.strings.regex_replace(image_path, ".*[/\\\\]", "")
        
        full_path = tf.strings.join([IMAGE_BASE_DIR, clean_image_path])
        image = load_image_tf(full_path)

        # Text (Python tokenizer via py_function)
        input_ids, attention_mask = tf.py_function(
            tokenize_py,
            [review_text],
            [tf.int32, tf.int32]
        )

        # IMPORTANT: set static shapes
        input_ids.set_shape([MAX_LEN])
        attention_mask.set_shape([MAX_LEN])

        return image, input_ids, attention_mask
    except Exception as e:
        logging.error(f"Error processing sample: {str(e)}")
        raise CustomException(e, sys)