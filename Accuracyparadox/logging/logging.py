import logging
import os
from datetime import datetime

# Use project-root logs folder (stable, not dependent on current working dir)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Export this and use it in components
logger = logging.getLogger("AccuracyParadox")