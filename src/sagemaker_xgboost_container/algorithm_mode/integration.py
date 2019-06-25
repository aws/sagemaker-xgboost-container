import logging
import logging.config


FORMATTERS = {
    'verbose': {
        'format': '[%(asctime)s:%(levelname)s] %(message)s',
        'datefmt': "%Y-%m-%d:%H:%M:%S",
    },
    'simple': {
        'format': '[%(levelname)s:%(name)s] %(message)s'
    },
}

CONSOLE_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': FORMATTERS,
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'verbose',
            'class': 'logging.StreamHandler',
            'stream': None
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    }
}


LOGGING_CONFIGS = {
    "console_only": CONSOLE_LOGGING,
}


def setup_main_logger(name):
    """
    Return a logger that configures logging for the main application.

    :param name: Name of the returned logger.
    """

    log_config = LOGGING_CONFIGS["console_only"]
    logging.config.dictConfig(log_config)
    return logging.getLogger(name)
