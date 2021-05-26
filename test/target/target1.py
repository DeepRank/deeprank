import numpy


def __compute_target__(decoy, target_group):
    target_group.create_dataset("target1", data=numpy.array(1.0))
