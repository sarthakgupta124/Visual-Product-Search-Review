import tensorflow as tf
from src.exception import CustomException
import sys
from src.logger import get_logger
logging = get_logger(__name__)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
from src.utils import get_training_variables
MAX_LEN = get_training_variables()['MAX_LEN']

def tokenize_py(text):
    try:
        tokens = tokenizer(
        text.numpy().decode("utf-8"),
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="np")
        return tokens["input_ids"][0], tokens["attention_mask"][0]
    except Exception as e:
        logging.error(f"Error tokenizing text: {str(e)}")
        raise CustomException(e, sys)