# -*- coding: utf-8 -*-
"""
Default DeepRank settings. 
"""

####################
# DEBUG            #
####################

DEBUG = True

###########
# LOGGING #
###########

# Default logging settings for DeepRank.
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'brief': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'filters': {
        'use_levels': {
            '()': 'deeprank.utils.log.useLevelsFilter',
        },
        'require_debug': {
            '()': 'deeprank.utils.log.requireDebugFilter',
        },
    },
    'handlers': {
        'stdout': {
            'level': 'INFO',
            'formatter': 'brief',
            'filters': ['use_levels([logging.INFO])'],
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
        'stderr': {
            'level': 'WARNING',
            'formatter': 'brief',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
        },
        'debug': {
            'level': 'DEBUG',
            'formatter': 'brief',
            'filters': ['require_debug', 'use_levels([logging.DEBUG])'],
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
        },
    },
    'loggers': {
        'deeprank': {
            'handlers': ['stdout', 'stderr', 'debug'],
            'level': 'NOTSET',
        },
    }
}
