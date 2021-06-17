from tempfile import mkdtemp
from shutil import rmtree
import os

import numpy
import h5py
from nose.tools import eq_, ok_

from deeprank.models.variant import PdbVariantSelection
from deeprank.operate import hdf5data


def test_group_name():
    variant1 = PdbVariantSelection("not/existent/pdb.1", 'A', 111, 'M', {'A': 'not/existent/pssm.A.1'})
    variant2 = PdbVariantSelection("not/existent/pdb.2", 'A', 22, 'W', {'A': 'not/existent/pssm.A.2'})

    ok_(hdf5data.get_variant_group_name(variant1) != hdf5data.get_variant_group_name(variant2))
    eq_(hdf5data.get_variant_group_name(variant1), hdf5data.get_variant_group_name(variant1))
    eq_(hdf5data.get_variant_group_name(variant2), hdf5data.get_variant_group_name(variant2))


def test_variant():
    start_variant = PdbVariantSelection("not/existent/pdb", 'A', 111, 'M', {'A': 'not/existent/pssm.A'})

    temp_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(temp_dir_path, "test.hdf5")
        with h5py.File(hdf5_path, 'w') as f5:
            group = f5.require_group("variant1")

            hdf5data.store_variant(group, start_variant)

            end_variant = hdf5data.load_variant(group)

            eq_(start_variant, end_variant)
    finally:
        rmtree(temp_dir_path)


def _numpy_eq(a1, a2):
    assert numpy.allclose(a1, a2), "{} ! = {}".format(a1, a2)

def test_grid():
    center = numpy.array([0.0, 0.0, 0.0])
    box_dimension = 30
    resolution = 1.0
    point_counts = numpy.array([box_dimension, box_dimension, box_dimension])
    resolutions = numpy.array([resolution, resolution, resolution])

    x_coords = numpy.linspace(center[0] - 0.5 * point_counts[0] * resolutions[0],
                              center[0] + 0.5 * point_counts[0] * resolutions[0],
                              point_counts[0])

    y_coords = numpy.linspace(center[1] - 0.5 * point_counts[1] * resolutions[1],
                              center[1] + 0.5 * point_counts[1] * resolutions[1],
                              point_counts[1])

    z_coords = numpy.linspace(center[2] - 0.5 * point_counts[2] * resolutions[2],
                              center[2] + 0.5 * point_counts[2] * resolutions[2],
                              point_counts[2])

    xgrid, ygrid, zgrid = numpy.meshgrid(y_coords, x_coords, z_coords)
    distances = numpy.sqrt((xgrid - center[0])**2 + (ygrid - center[1])**2 + (zgrid - center[2])**2)
    feature_data = numpy.zeros(point_counts)
    feature_data = 1.0 * numpy.exp(-0.01 * distances)

    feature_name = "feature1"
    subfeature_name = "featureA"

    temp_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(temp_dir_path, "test.hdf5")

        with h5py.File(hdf5_path, 'w') as f5:
            group = f5.require_group("variant2")

            hdf5data.store_grid_points(group, x_coords, y_coords, z_coords)
            hdf5data.store_grid_center(group, center)
            hdf5data.store_grid_data(group, feature_name, {subfeature_name: feature_data})

            loaded_x_coords, loaded_y_coords, loaded_z_coords = hdf5data.load_grid_points(group)
            loaded_center = hdf5data.load_grid_center(group)
            loaded_feature_data = hdf5data.load_grid_data(group, feature_name)[subfeature_name]

            _numpy_eq(loaded_x_coords, x_coords)
            _numpy_eq(loaded_y_coords, y_coords)
            _numpy_eq(loaded_z_coords, z_coords)

            _numpy_eq(loaded_center, center)

            _numpy_eq(loaded_feature_data, feature_data)
    finally:
        rmtree(temp_dir_path)
