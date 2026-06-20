import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    raw_data_path: str = os.path.join('notebook', 'data', 'amazon_multimodal_dataset_with_ratings.csv')
    artifacts_dir: str = 'artifacts'
    # ingested_data_path: str = os.path.join('artifacts', 'ingested_data.csv')
    ingestion_data_path: str = os.path.join('artifacts', 'ingested_data.csv')
    eval_data_path: str = os.path.join('artifacts', 'eval_data.csv')
    sample_size: int = None  # None = use full cleaned dataset
    eval_size: int = 150


class DataIngestion:
    
    def __init__(self, config: DataIngestionConfig = None):
        self.config = DataIngestionConfig()
    
    def load_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from {self.config.raw_data_path}")
            df = pd.read_csv(self.config.raw_data_path)
            logger.info(f"Data shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            raise CustomException(e, sys)
        
    def clean_data(self, df):
        try:
            logger.info("Cleaning data")
            df=df.dropna(subset=["review_text"])
            df_cleaned = df.drop_duplicates()
            logger.info(f"Cleaned data shape: {df_cleaned.shape}")
            return df_cleaned
        except Exception as e:
            logger.error(f"Error in clean_data: {str(e)}")
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self) -> str:
        """
        Orchestrate data ingestion pipeline.
        
        1. Loads raw data from CSV
        2. Cleans data (removes NaN, duplicates)
        3. Holds out eval set (150 samples, no leakage)
        4. Saves training set as ingested_data.csv
        5. Saves eval set as eval_data.csv
        
        Returns:
            Path to ingested_data.csv (training set)
        """
        try:
            df = self.load_data()
            df_cleaned = self.clean_data(df)

            logger.info(f"Cleaned dataset size: {len(df_cleaned)}")
            logger.info(f"Holding out {self.config.eval_size} samples for evaluation")
            
            # Hold out the eval set FIRST so it never leaks into training
            df_eval = df_cleaned.sample(n=self.config.eval_size, random_state=123)
            df_remaining = df_cleaned.drop(df_eval.index)

            # Use full remaining data if sample_size is None, otherwise sample
            df_train = (
                df_remaining if self.config.sample_size is None
                else df_remaining.sample(n=self.config.sample_size, random_state=42)
            )
            
            logger.info(f"Training set size: {len(df_train)}")
            logger.info(f"Eval set size: {len(df_eval)}")
            
            os.makedirs(self.config.artifacts_dir, exist_ok=True)
            df_train.to_csv(self.config.ingestion_data_path, index=False, header=True)
            df_eval.to_csv(self.config.eval_data_path, index=False, header=True)
            
            logger.info(f"Saved training data to: {self.config.ingestion_data_path}")
            logger.info(f"Saved eval data to: {self.config.eval_data_path}")
            
            return self.config.ingestion_data_path
        except Exception as e:
            logger.error(f"Error in initiate_data_ingestion: {str(e)}")
            raise CustomException(e, sys)
