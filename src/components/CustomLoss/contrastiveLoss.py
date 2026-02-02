from src.logger import get_logger
logging = get_logger(__name__)
import tensorflow as tf

class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def call(self, img_emb, txt_emb):
        try:
            logits = tf.matmul(img_emb, txt_emb, transpose_b=True)
            logits /= self.temperature
            labels = tf.range(tf.shape(logits)[0])
            return tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    labels, logits, from_logits=True
                )
            )
        except Exception as e:
            logging.error(f"Error in ContrastiveLoss call: {str(e)}")
            raise e