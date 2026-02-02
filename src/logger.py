import logging
import os
import sys
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Create log file with timestamp
log_file = os.path.join(logs_dir, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure logging with UTF-8 encoding for console handler
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Stream handler with UTF-8 encoding
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
stream_handler.stream.reconfigure(encoding='utf-8') if hasattr(stream_handler.stream, 'reconfigure') else None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)

def get_logger(name: str):
    """Get logger instance for a module"""
    return logging.getLogger(name)
