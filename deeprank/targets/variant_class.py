import numpy


def __compute_target__(variant, target_group):
    if variant.variant_class is None:
        raise ValueError("class isn't set on mutant {}".format(variant))

    target_group.create_dataset("class", data=numpy.array(float(variant.variant_class.value)))
