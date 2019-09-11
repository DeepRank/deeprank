import logging
import logging.config

from . import logger_settings
from .chemicals import AA_codes, AA_codes_3to1, AA_codes_1to3
from .chemicals import AA_codes_pssm_ordered
from .chemicals import AA_properties

# Debug
DEBUG = False

# Default logger
logging.config.dictConfig(logger_settings.DEFAULT_LOGGING)
logger = logging.getLogger('deeprank')

# Default PSSM path
PATH_PSSM_SOURCE = None
