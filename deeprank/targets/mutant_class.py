import numpy


def __compute_target__(mutant, target_group):
    if mutant.mutant_class is None:
        raise ValueError("class isn't set on mutant {}".format(mutant))

    target_group.create_dataset("class", data=numpy.array(float(mutant.mutant_class.value)))
