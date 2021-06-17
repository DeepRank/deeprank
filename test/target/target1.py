import numpy

from deeprank.models.variant import PdbVariantSelection


def __compute_target__(variant, target_group):
    if type(variant) != PdbVariantSelection:
        raise TypeError("PdbVariantSelection expected, got {}".format(type(variant)))

    target_group.create_dataset("target1", data=numpy.array(1.0))
