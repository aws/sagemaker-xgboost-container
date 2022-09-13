# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import logging
import logging.config

FORMATTERS = {
    "verbose": {
        "format": "[%(asctime)s:%(levelname)s] %(message)s",
        "datefmt": "%Y-%m-%d:%H:%M:%S",
    },
    "simple": {"format": "[%(levelname)s:%(name)s] %(message)s"},
}

CONSOLE_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": FORMATTERS,
    "handlers": {
        "console": {"level": "INFO", "formatter": "verbose", "class": "logging.StreamHandler", "stream": None},
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
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
