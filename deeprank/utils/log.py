import logging
from deeprank import global_settings


class useLevelsFilter(logging.Filter):
    def __init__(self, levels):
        if not isinstance(levels, (tuple, list)):
            levels = (levels, )
        self.levels = levels

    def __filter__(self, record):
        if record.levelname in self.levels:
            return True


class requireDebugFilter(logging.Filter):
    def __filter__(self, record):
        return global_settings.DEBUG
