import re
import logging


_log = logging.getLogger(__name__)

class VanderwaalsParam:
    def __init__(self, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma


class ParamParser:

    LINE_PATTERN = re.compile(r"^NONBonded\s+([A-Z0-9]{1,4})((\s+\-?[0-9]+\.[0-9]+){4})\s*$")

    @staticmethod
    def parse(file_):
        result = {}
        for line in file_:
            if line.startswith('#') or len(line.strip()) == 0:
                continue

            m = ParamParser.LINE_PATTERN.match(line)
            if not m:
                raise ValueError("unmatched param line: {}".format(repr(line)))

            atom_type = m.group(1)
            if atom_type in result:
                raise ValueError("duplicate atom type: {}".format(repr(atom_type)))

            number_strings = m.group(2).split()
            epsilon = float(number_strings[0])
            sigma = float(number_strings[1])

            result[atom_type] = VanderwaalsParam(epsilon, sigma)
        return result
