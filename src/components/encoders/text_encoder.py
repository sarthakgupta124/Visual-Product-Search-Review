import sys
import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras import layers, models
from src.exception import CustomException
from src.logger import get_logger
from src.utils import get_training_variables

logging = get_logger(__name__)
MAX_LEN = get_training_variables()['MAX_LEN']

# 1. Create a custom layer to track BERT variables
# class BertEmbeddingLayer(layers.Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         # Loading here and assigning to self ensures Keras tracks the weights
#         self.bert = TFBertModel.from_pretrained("bert-base-uncased")
#         self.bert.trainable = True
#         for layer in self.bert.layers[0].encoder.layer[:-2]:
#             layer.trainable = False

#     def call(self, inputs):
#         input_ids, attention_mask = inputs
#         # BERT returns a TFBaseModelOutputWithPooling object
#         out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         # pooler_output is the [CLS] token after a dense + tanh layer
#         return out.pooler_output

# class TextEncoder:
#     def __init__(self, embed_dim=256):
#         self.embed_dim = embed_dim
#         self.model = self.build_text_encoder()

#     def build_text_encoder(self):
#         try:
#             # 2. Define Inputs
#             input_ids = layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
#             attention_mask = layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")

#             # 3. Use the custom layer (No more Lambda for BERT call)
#             cls_emb = BertEmbeddingLayer()([input_ids, attention_mask])

#             # 4. Projection and Normalization
#             x = layers.Dense(self.embed_dim, name="projection")(cls_emb)
            
#             # Simple math Lambdas are usually fine for saving
#             outputs = layers.Lambda(
#                 lambda t: tf.math.l2_normalize(t, axis=1),
#                 name="l2_norm"
#             )(x)

#             return models.Model(inputs=[input_ids, attention_mask], outputs=outputs)
            
#         except Exception as e:
#             logging.error(f"Error in text encoding: {str(e)}")
#             raise CustomException(e, sys)

# src/components/encoders/text_encoder.py
class BertEmbeddingLayer(layers.Layer):
    def __init__(self, fine_tune_last_n_layers: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained("bert-base-uncased")
        if fine_tune_last_n_layers > 0:
            self.bert.trainable = True
            for layer in self.bert.layers[0].encoder.layer[:-fine_tune_last_n_layers]:
                layer.trainable = False
        else:
            self.bert.trainable = False

    def call(self, inputs):
        input_ids, attention_mask = inputs
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output

class TextEncoder:
    def __init__(self, embed_dim=256, fine_tune_last_n_layers: int = 0):
        self.embed_dim = embed_dim
        self.fine_tune_last_n_layers = fine_tune_last_n_layers
        self.model = self.build_text_encoder()

    def build_text_encoder(self):
        try:
            input_ids = layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
            attention_mask = layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")
            cls_emb = BertEmbeddingLayer(fine_tune_last_n_layers=self.fine_tune_last_n_layers)(
                [input_ids, attention_mask]
            )
            x = layers.Dense(self.embed_dim, name="projection")(cls_emb)
            outputs = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="l2_norm")(x)
            return models.Model(inputs=[input_ids, attention_mask], outputs=outputs)
        except Exception as e:
            logging.error(f"Error in text encoding: {str(e)}")
            raise CustomException(e, sys)