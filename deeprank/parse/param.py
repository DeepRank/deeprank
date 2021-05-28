import re
import logging

from deeprank.models.param import VanderwaalsParam

_log = logging.getLogger(__name__)


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

            inter_epsilon = float(number_strings[0])
            inter_sigma = float(number_strings[1])
            intra_epsilon = float(number_strings[2])
            intra_sigma = float(number_strings[3])

            result[atom_type] = VanderwaalsParam(inter_epsilon, inter_sigma,
                                                 intra_epsilon, intra_sigma)
        return result
