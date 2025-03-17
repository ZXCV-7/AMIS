import os
import logging
import datetime

def setup_logger(filename='logfile.log'):
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    full_filename = os.path.join(log_dir, f'{timestamp}_{filename}')

    log_formatter = logging.Formatter('%(asctime)s: %(message)s')
    logger = logging.getLogger('shared_logger')
    logger.setLevel(logging.INFO)

    logger.propagate = False
    logger.handlers.clear()

    f_handler = logging.FileHandler(full_filename, mode='w')
    f_handler.setFormatter(log_formatter)
    logger.addHandler(f_handler)

    c_handler = logging.StreamHandler()
    c_handler.setFormatter(log_formatter)
    logger.addHandler(c_handler)
    logging.getLogger().handlers.clear()
    return logger
