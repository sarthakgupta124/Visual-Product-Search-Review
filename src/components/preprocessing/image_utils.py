import tensorflow as tf
from src.exception import CustomException
import sys
from src.logger import get_logger
logging = get_logger(__name__)
def load_image(path):
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        return img
    except Exception as e:
        logging.error(f"Error loading image: {str(e)}")
        raise CustomException(e, sys)

