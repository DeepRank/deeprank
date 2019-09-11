# Default logging settings for DeepRank.
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'brief': {
            'format': '%(message)s',
        },
        'precise': {
            'format': '%(levelname)s: %(message)s',
        },
    },
    'filters': {
        'require_debug': {
            '()': 'deeprank.utils.logger_helper.requireDebugFilter',
        },
        'use_only_info_level': {
            '()': 'deeprank.utils.logger_helper.useLevelsFilter',
            'levels': 'INFO',   # function parameter
        },
        'use_only_debug_level': {
            '()': 'deeprank.utils.logger_helper.useLevelsFilter',
            'levels': 'DEBUG',
        },
    },
    'handlers': {
        'stdout': {
            'level': 'INFO',
            'formatter': 'brief',
            'filters': ['use_only_info_level', ],
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
        'stderr': {
            'level': 'WARNING',
            'formatter': 'precise',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
        },
        'debug': {
            'level': 'DEBUG',
            'formatter': 'precise',
            'filters': ['require_debug', 'use_only_debug_level'],
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
        },
    },
    'loggers': {
        'deeprank': {
            'handlers': ['stdout', 'stderr', 'debug'],
            'level': 'DEBUG',
        },
    }
}