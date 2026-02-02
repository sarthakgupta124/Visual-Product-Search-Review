from src.components.CustomLoss.contrastiveLoss import ContrastiveLoss
from src.logger import get_logger
logging = get_logger(__name__)
import tensorflow as tf
from src.utils import get_training_variables, get_custom_paths
from src.exception import CustomException
import sys
import pandas as pd
from src.components.encoders.text_encoder import TextEncoder
from src.components.encoders.image_encoder import ImageEncoder
from src.components.data_processing.sample_preprocessing import process_sample
from src.components.data_ingestion import DataIngestion

class Trainer:
    def __init__(self):
        try:
            self.paths = get_custom_paths()
            self.params = get_training_variables()
            self.text_encoder = TextEncoder()
            self.image_encoder = ImageEncoder()
            self.optimizer = tf.keras.optimizers.Adam(1e-4)
            self.loss_fn = ContrastiveLoss()
            self.process_sample = process_sample
            self.data_ingestion = DataIngestion()
        except Exception as e:
            logging.error(f"Error in Trainer initialization: {str(e)}")
            raise CustomException(e, sys)
    def train_step(self,images, input_ids, attention_mask):
        try:
            with tf.GradientTape() as tape:
                # Reference the .model attribute created in your Encoder classes
                img_emb = self.image_encoder.model(images, training=True)
                txt_emb = self.text_encoder.model([input_ids, attention_mask], training=False)
                loss = self.loss_fn(img_emb, txt_emb)

            vars = self.image_encoder.model.trainable_variables
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))
            return loss
        except Exception as e:
            logging.error(f"Error in train_step: {str(e)}")
            raise CustomException(e, sys)
        
    def checkpoint_models(self):
        try:
            checkpoint = tf.train.Checkpoint(
            image_encoder=self.image_encoder.model,
            text_encoder=self.text_encoder.model,
            optimizer=self.optimizer
            )
            ckpt_manager = tf.train.CheckpointManager(
            checkpoint,
            self.paths['CKPT_DIR'],
            max_to_keep=5
            )
            if ckpt_manager.latest_checkpoint:
                checkpoint.restore(ckpt_manager.latest_checkpoint)
                print("✅ Restored from:", ckpt_manager.latest_checkpoint)
            else:
                print("🆕 Training from scratch")


            logging.info("Models checkpointed successfully.")       
            save_path = ckpt_manager.save()
            return save_path  # ✅ ADD THIS LINE
        except Exception as e:
            logging.error(f"Error in checkpoint_models: {str(e)}")
            raise CustomException(e, sys)
    
    def train(self, train_dataset):
        try:
            for epoch in range(self.params['EPOCHS']):
                print(f"\nEpoch {epoch+1}/{self.params['EPOCHS']}")

                for images, input_ids, attention_mask in train_dataset:
                    loss = self.train_step(images, input_ids, attention_mask)

                save_path = self.checkpoint_models()
                print(f"💾 Saved checkpoint: {save_path}")
                print("Epoch loss:", loss.numpy())
            logging.info("Training completed successfully.")
        except Exception as e:
            logging.error(f"Error in train: {str(e)}")
            raise CustomException(e, sys)
        
    def save_models(self):
        try:
            # Save as a TensorFlow SavedModel directory
            self.text_encoder.model.save(self.paths['TEXT_ENCODER_PATH'], save_format='tf')
            self.image_encoder.model.save(self.paths['IMAGE_ENCODER_PATH'], save_format='tf')
            logging.info("Models saved successfully in SavedModel format.")
        except Exception as e:
            logging.error(f"Error in save_models: {str(e)}")
            raise CustomException(e, sys)
        
    def initialize_training(self):
        try:
            self.data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed successfully.")
            df= pd.read_csv(self.paths['ingestion_data_path'])
            logging.info(f"Loaded ingested data shape: {df.shape}")

            dataset = tf.data.Dataset.from_tensor_slices(
                (df["image_path"].values, df["review_text"].values)
            )

            dataset = (
                dataset
                .shuffle(1000)
                .map(self.process_sample, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(self.params['BATCH_SIZE'])
                .prefetch(tf.data.AUTOTUNE)
            )

            self.checkpoint_models()
            self.train(dataset)
            self.save_models()
            logging.info("Training initialization completed successfully.")
        except Exception as e:
            logging.error(f"Error in initialize_training: {str(e)}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    try:
        trainer = Trainer()
        trainer.initialize_training()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise CustomException(e, sys)