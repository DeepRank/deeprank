import logging
import logging.config

from . import logger_settings

# Debug
DEBUG = False

# Default logger
logging.config.dictConfig(logger_settings.DEFAULT_LOGGING)
logger = logging.getLogger('deeprank')

# Default PSSM path
PATH_PSSM_SOURCE = None