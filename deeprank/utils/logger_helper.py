import logging

import deeprank.config


class useLevelsFilter(logging.Filter):
    def __init__(self, levels):
        if not isinstance(levels, (tuple, list)):
            levels = (levels, )
        self.levelnos = [getattr(logging, i) for i in levels]

    def filter(self, record):
        if record.levelno in self.levelnos:
            return True


class requireDebugFilter(logging.Filter):
    def filter(self, record):
        return deeprank.config.DEBUG