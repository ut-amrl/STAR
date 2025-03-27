import logging
import os
from datetime import datetime

def get_logger(name="agent", log_dir="debug/agent"):
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding multiple handlers in interactive environments
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger