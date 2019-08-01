import logging
import logging.config

from . import global_settings

logging.config.dictConfig(global_settings.DEFAULT_LOGGING)
logger = logging.getLogger('deeprank')
