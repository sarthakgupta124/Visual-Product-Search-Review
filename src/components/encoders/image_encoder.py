from src.exception import CustomException
from src.logger import get_logger
logging=get_logger(__name__)
import sys
from src.utils import get_training_variables
MAX_LEN = get_training_variables()['MAX_LEN']
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
# class ImageEncoder:
#     def __init__(self, embed_dim=256):  # ✅ Accept parameters in __init__
#         self.embed_dim = embed_dim
#         self.model = self.build_image_encoder()
    
#     def build_image_encoder(self,fine_tune_last_n_layers: int = 0):  # ✅ Instance method with 'self'
#         try:
#             # inputs = layers.Input(shape=(224, 224, 3))
#             # base = ResNet50(
#             #     weights="imagenet",
#             #     include_top=False,
#             #     input_tensor=inputs
#             # )
#             # base.trainable = False
#             # x = base.output
#             # x = layers.GlobalAveragePooling2D()(x)
#             # x = layers.Dense(self.embed_dim)(x)
#             # outputs = layers.Lambda(
#             #     lambda t: tf.math.l2_normalize(t, axis=1),
#             #     output_shape=(self.embed_dim,)
#             # )(x)
#             # return models.Model(inputs, outputs)
#             inputs = layers.Input(shape=(224, 224, 3))
#             base = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

#             if fine_tune_last_n_layers > 0:
#                 base.trainable = True
#                 for layer in base.layers[:-fine_tune_last_n_layers]:
#                     layer.trainable = False
#             else:
#                 base.trainable = False

#             x = base.output
#             x = layers.GlobalAveragePooling2D()(x)
#             x = layers.Dense(self.embed_dim)(x)
#             outputs = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
#             return models.Model(inputs, outputs)
#         except Exception as e:
#             logging.error(f"Error in image encoding: {str(e)}")
#             raise CustomException(e, sys)
# src/components/encoders/image_encoder.py
class ImageEncoder:
    def __init__(self, embed_dim=256, fine_tune_last_n_layers: int = 0):
        self.embed_dim = embed_dim
        self.model = self.build_image_encoder(fine_tune_last_n_layers)

    def build_image_encoder(self, fine_tune_last_n_layers: int = 0):
        try:
            inputs = layers.Input(shape=(224, 224, 3))
            base = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
            if fine_tune_last_n_layers > 0:
                base.trainable = True
                for layer in base.layers[:-fine_tune_last_n_layers]:
                    layer.trainable = False
            else:
                base.trainable = False
            x = base.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(self.embed_dim)(x)
            outputs = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
            return models.Model(inputs, outputs)
        except Exception as e:
            logging.error(f"Error in image encoding: {str(e)}")
            raise CustomException(e, sys)
    
    