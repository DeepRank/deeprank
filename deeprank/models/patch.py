from enum import Enum


class PatchActionType(Enum):
    MODIFY = 1
    ADD = 2


class PatchResidueSelectionType(Enum):
    NTER = 1
    PROP = 2
    CTER = 3
    CTN = 4
    DISU = 5
    CYNH = 6
    HISE = 7
    HISD = 8
