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
    ingested_data_path: str = os.path.join('artifacts', 'ingested_data.csv')


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
        try:
            logger.info("Starting data ingestion")
            
            # Load raw data
            df = self.load_data()
            
            df_cleaned = self.clean_data(df)
            # Save ingested data
            os.makedirs(os.path.dirname(self.config.ingested_data_path), exist_ok=True)
            df_sampled = df_cleaned.sample(n=1000, random_state=42)
            df_sampled.to_csv(self.config.ingested_data_path, index=False,header=True)
            
            logger.info("Data ingestion completed successfully")
            return self.config.ingested_data_path
        except Exception as e:
            logger.error(f"Error in initiate_data_ingestion: {str(e)}")
            raise CustomException(e, sys)
