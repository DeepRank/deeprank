import re


class TopRowObject:
    def __init__(self, residue_name, atom_name, kwargs):
        self.residue_name = residue_name
        self.atom_name = atom_name
        self.kwargs = kwargs


class TopParser:
    VAR_PATTERN = re.compile(r"([^\s]+)\s*=\s*([^\s\(\)]+|\(.*\))")
    LINE_PATTERN = re.compile(
        r"^([A-Z0-9]{3})\s+atom\s+([A-Z0-9]{1,4})\s+(.+)\s+end\s*(\s+\!\s+[ _A-Za-z0-9]+)?$")
    NUMBER_PATTERN = re.compile(r"\-?[0-9]+(\.[0-9]+)?")

    @staticmethod
    def parse(file_):
        result = []
        for line in file_:
            # parse the line
            m = TopParser.LINE_PATTERN.match(line)
            if not m:
                raise ValueError("Unmatched top line: {}".format(repr(line)))

            residue_name = m.group(1).upper()
            atom_name = m.group(2).upper()

            kwargs = {}
            for w in TopParser.VAR_PATTERN.finditer(m.group(3)):
                kwargs[w.group(1).lower().strip()] = TopParser._parse_value(w.group(2).strip())

            result.append(TopRowObject(residue_name, atom_name, kwargs))

        return result

    @staticmethod
    def _parse_value(s):
        # remove parentheses
        if s[0] == '(' and s[-1] == ')':
            return TopParser._parse_value(s[1:-1])

        if TopParser.NUMBER_PATTERN.match(s):
            return float(s)
        else:
            return s
