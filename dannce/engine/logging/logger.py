import logging
import logging.config
from pathlib import Path
import json
from collections import OrderedDict
import os

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

log_levels = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}

log_config = OrderedDict({
    "version": 1, 
    "disable_existing_loggers": False, 
    "formatters": {
        "simple": {"format": "%(message)s"}, 
        "datetime": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    }, 
    "handlers": {
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler", 
            "level": "INFO", 
            "formatter": "datetime", 
            "filename": "training.log", 
            "maxBytes": 10485760, 
            "backupCount": 20, "encoding": "utf8"
        }
    }, 
    "root": {
        "level": "INFO", 
        "handlers": [
            "info_file_handler"
        ]
    }
})

def setup_logging(save_dir, filename="training.log"):
    """
    Setup logging configuration
    """
    for _, handler in log_config['handlers'].items():
        if 'filename' in handler:
            handler['filename'] = os.path.join(save_dir, filename)
        logging.config.dictConfig(log_config)


def get_logger(verbosity=2):
    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger()
    logger.setLevel(log_levels[verbosity])
    return logger