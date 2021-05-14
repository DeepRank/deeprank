import numpy

from deeprank.models.mutant import PdbMutantSelection
from deeprank.tools import sparse


def store_mutant(mutant_group, mutant):
    """ Stores the mutant in the HDF5 mutant group

        Args:
            mutant_group (HDF5 group): the group belonging to the mutant selection
            mutant (PdbMutantSelection): the mutant object
    """

    mutant_group.attrs['pdb_path'] = mutant.pdb_path

    for chain_id in mutant.get_pssm_chains():
        mutant_group.attrs['pssm_path_%s' % chain_id] = mutant.get_pssm_path(chain_id)

    mutant_group.attrs['mutant_chain_id'] = mutant.chain_id

    mutant_group.attrs['mutant_residue_number'] = mutant.residue_number

    mutant_group.attrs['mutant_amino_acid'] = mutant.mutant_amino_acid


def load_mutant(mutant_group):
    """ Loads the mutant from the HDF5 mutant group

        Args:
            mutant_group (HDF5 group): the group belonging to the mutant selection

        Returns (PdbMutantSelection): the mutant object
    """

    pdb_path = mutant_group.attrs['pdb_path']

    pssm_paths_by_chain = {}
    for attr_name in mutant_group.attrs:
        if attr_name.startswith("pssm_path_"):
            chain_id = attr_name.split('_')[-1]
            pssm_paths_by_chain[chain_id] = mutant_group.attrs[attr_name]

    chain_id = mutant_group.attrs['mutant_chain_id']

    residue_number = mutant_group.attrs['mutant_residue_number']

    amino_acid = mutant_group.attrs['mutant_amino_acid']

    mutant = PdbMutantSelection(pdb_path, chain_id, residue_number, amino_acid, pssm_paths_by_chain)

    return mutant


def store_grid_center(mutant_group, center):
    """ Stores the center position in the HDF5 mutant group

        Args:
            mutant_group (HDF5 group): the group belonging to the mutant selection
            center (float, float, float): xyz position of the center
    """

    grid_group = mutant_group.require_group("grid_points")

    grid_group.create_dataset('center', data=center)


def load_grid_center(mutant_group):
    """ Loads the center position from the HDF5 mutant group

        Args:
            mutant_group (HDF5 group): the group belonging to the mutant selection

        Returns (float, float, float): xyz position of the center
    """

    grid_group = mutant_group['grid_points']

    return numpy.array(grid_group['center'])


def store_grid_points(mutant_group, x_coords, y_coords, z_coords):
    """ Stores the grid point coordinates in the HDF5 mutant group

        Args:
            mutant_group (HDF5 group): the group belonging to the mutant selection
            x_coords (list(float)): the x coords of the grid points
            y_coords (list(float)): the y coords of the grid points
            z_coords (list(float)): the z coords of the grid points
    """

    grid_group = mutant_group.require_group("grid_points")

    grid_group.create_dataset('x', data=x_coords)
    grid_group.create_dataset('y', data=y_coords)
    grid_group.create_dataset('z', data=z_coords)


def load_grid_points(mutant_group):
    """ Loads the grid point coordinates from the HDF5 mutant group

        Args:
            mutant_group (HDF5 group): the group belonging to the mutant selection

        Returns (list(float), list(float), list(float)): the x, y and z coordinates of the grid points
    """

    grid_group = mutant_group['grid_points']

    x_coords = numpy.array(grid_group['x'])
    y_coords = numpy.array(grid_group['y'])
    z_coords = numpy.array(grid_group['z'])

    return (x_coords, y_coords, z_coords)


def store_grid_data(mutant_group, feature_name, feature_dict):
    """ Store 3D grid data in a mutant group.

        Args:
            mutant_group (HDF5 group): the group belonging to the mutant selection
            feature_name (str): the name of the feature to store
            feature_dict (dict(str, matrix(number))): a dictionary, containing the data per subfeature name
    """

    feature_group = mutant_group.require_group("mapped_features/%s" % feature_name)

    for subfeature_name, subfeature_data in feature_dict.items():

        # Remove the old data (if present).
        if subfeature_name in feature_group:
            del(feature_group[subfeature_name])

        # Create the subfeature group anew.
        subfeature_group = feature_group.create_group(subfeature_name)

        spg = sparse.FLANgrid()
        spg.from_dense(subfeature_data, beta=1E-2)

        if spg.sparse:
            subfeature_group.attrs['sparse'] = True
            subfeature_group.attrs['type'] = 'sparse_matrix'
            subfeature_group.create_dataset('index', data=spg.index, compression='gzip', compression_opts=9)
            subfeature_group.create_dataset('value', data=spg.value, compression='gzip', compression_opts=9)
        else:
            subfeature_group.attrs['sparse'] = False
            subfeature_group.attrs['type'] = 'dense_matrix'
            subfeature_group.create_dataset('value', data=spg.value, compression='gzip', compression_opts=9)


def load_grid_data(mutant_group, feature_name):
    """ Load 3D grid data from a mutant group

        Args:
            mutant_group (HDF5 group): the group belonging to the mutant selection
            feature_name (str): the name of the feature to store

        Returns (dict(str, matrix(number))): a dictionary, containing the data per subfeature name
    """

    grid_shape = numpy.shape(load_grid_points(mutant_group))

    feature_group = mutant_group["mapped_features/%s" % feature_name]

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
