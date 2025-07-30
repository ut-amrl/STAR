import logging
import os
from datetime import datetime

def get_logger(name="agent", logdir="debug/agent", prefix: str = "", flatten: bool = False):
    os.makedirs(logdir, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    if not flatten:
        logdir = os.path.join(logdir, date_str)
        os.makedirs(logdir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if prefix:
        log_file = os.path.join(logdir, f"{prefix}_{timestamp}.log")
    else:
        log_file = os.path.join(logdir, f"{timestamp}.log")

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