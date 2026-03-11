import logging
import os
from datetime import datetime

## Log file name with timestamp

LOG_FILE = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

## Log file path
log_path = os.path.join(os.getcwd(), 'logs')

## Create logs directory if it doesn't exist
os.makedirs(log_path, exist_ok=True)
## Full log file path
log_file_path = os.path.join(log_path, LOG_FILE)

## Configure logging
logging.basicConfig(
    filename=log_file_path,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)