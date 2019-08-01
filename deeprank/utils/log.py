import logging
from deeprank import global_settings


class useLevelsFilter(logging.Filter):
    def __init__(self, *args):
        self.levels = args

    def __filter__(self, record):
        if record.levelname in self.levels:
            return True


class requireDebugFilter(logging.Filter):
    def __filter__(self, record):
        return global_settings.DEBUG
