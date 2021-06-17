import os

import numpy

from deeprank.models.variant import PdbVariantSelection
from deeprank.tools import sparse


def get_variant_group_name(variant):
    """
        Args:
            variant (PdbVariantSelection): a variant object
        Returns (str): an unique name for a given variant object
    """

    mol_name = os.path.splitext(os.path.basename(variant.pdb_path))[0]

    return "%s-%s" % (mol_name, str(hash(variant)).replace('-', 'm'))


def store_variant(variant_group, variant):
    """ Stores the variant in the HDF5 variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection
            variant (PdbVariantSelection): the variant object
    """

    variant_group.attrs['pdb_path'] = variant.pdb_path

    for chain_id in variant.get_pssm_chains():
        variant_group.attrs['pssm_path_%s' % chain_id] = variant.get_pssm_path(chain_id)

    variant_group.attrs['variant_chain_id'] = variant.chain_id

    variant_group.attrs['variant_residue_number'] = variant.residue_number

    variant_group.attrs['variant_amino_acid'] = variant.amino_acid


def load_variant(variant_group):
    """ Loads the variant from the HDF5 variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection

        Returns (PdbVariantSelection): the variant object
    """

    pdb_path = variant_group.attrs['pdb_path']

    pssm_paths_by_chain = {}
    for attr_name in variant_group.attrs:
        if attr_name.startswith("pssm_path_"):
            chain_id = attr_name.split('_')[-1]
            pssm_paths_by_chain[chain_id] = variant_group.attrs[attr_name]

    chain_id = variant_group.attrs['variant_chain_id']

    residue_number = variant_group.attrs['variant_residue_number']

    amino_acid = variant_group.attrs['variant_amino_acid']

    variant = PdbVariantSelection(pdb_path, chain_id, residue_number, amino_acid, pssm_paths_by_chain)

    return variant


def store_grid_center(variant_group, center):
    """ Stores the center position in the HDF5 variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection
            center (float, float, float): xyz position of the center
    """

    grid_group = variant_group.require_group("grid_points")

    if 'center' in grid_group:
        del(grid_group['center'])

    grid_group.create_dataset('center', data=center)


def load_grid_center(variant_group):
    """ Loads the center position from the HDF5 variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection

        Returns (float, float, float): xyz position of the center
    """

    grid_group = variant_group['grid_points']

    return numpy.array(grid_group['center'])


def store_grid_points(variant_group, x_coords, y_coords, z_coords):
    """ Stores the grid point coordinates in the HDF5 variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection
            x_coords (list(float)): the x coords of the grid points
            y_coords (list(float)): the y coords of the grid points
            z_coords (list(float)): the z coords of the grid points
    """

    grid_group = variant_group.require_group("grid_points")

    for coord in ['x', 'y', 'z']:
        if coord in grid_group:
            del(grid_group[coord])

    grid_group.create_dataset('x', data=x_coords)
    grid_group.create_dataset('y', data=y_coords)
    grid_group.create_dataset('z', data=z_coords)


def load_grid_points(variant_group):
    """ Loads the grid point coordinates from the HDF5 variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection

        Returns (list(float), list(float), list(float)): the x, y and z coordinates of the grid points
    """

    grid_group = variant_group['grid_points']

    x_coords = numpy.array(grid_group['x'])
    y_coords = numpy.array(grid_group['y'])
    z_coords = numpy.array(grid_group['z'])

    return (x_coords, y_coords, z_coords)


def store_grid_data(variant_group, feature_name, feature_dict, try_sparse=True):
    """ Store 3D grid data in a variant group.

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection
            feature_name (str): the name of the feature to store
            feature_dict (dict(str, matrix(number))): a dictionary, containing the data per subfeature name
    """

    feature_group = variant_group.require_group("mapped_features/%s" % feature_name)

    for subfeature_name, subfeature_data in feature_dict.items():

        # Remove the old data (if present).
        if subfeature_name in feature_group:
            del(feature_group[subfeature_name])

        # Create the subfeature group anew.
        subfeature_group = feature_group.create_group(subfeature_name)

        if try_sparse:
            spg = sparse.FLANgrid()
            spg.from_dense(subfeature_data, beta=1E-2)

        if try_sparse and spg.sparse:
            subfeature_group.attrs['sparse'] = True
            subfeature_group.attrs['type'] = 'sparse_matrix'
            subfeature_group.create_dataset('index', data=spg.index, compression='gzip', compression_opts=9)
            subfeature_group.create_dataset('value', data=spg.value, compression='gzip', compression_opts=9)
        else:
            subfeature_group.attrs['sparse'] = False
            subfeature_group.attrs['type'] = 'dense_matrix'
            subfeature_group.create_dataset('value', data=subfeature_data, compression='gzip', compression_opts=9)


def load_grid_data(variant_group, feature_name):
    """ Load 3D grid data from a variant group

        Args:
            variant_group (HDF5 group): the group belonging to the variant selection
            feature_name (str): the name of the feature to store

        Returns (dict(str, matrix(number))): a dictionary, containing the data per subfeature name
    """

    grid_shape = numpy.shape(load_grid_points(variant_group))

    feature_group = variant_group["mapped_features/%s" % feature_name]

    grid_data = {}
    for subfeature_name in feature_group.keys():
        subfeature_group = feature_group[subfeature_name]

        if subfeature_group.attrs['sparse']:

            spg = sparse.FLANgrid(True,
                                  subfeature_group['index'], subfeature_group['value'],
                                  grid_shape)
            grid_data[subfeature_name] = numpy.array(spg.to_dense())
        else:
            grid_data[subfeature_name] = numpy.array(subfeature_group['value'])

    return grid_data
