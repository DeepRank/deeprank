import os
import pickle

import h5py
import numpy as np

from deeprank.tools import sparse


class NormalizeData(object):

    def __init__(self, fname, shape=None):
        """Compute the normalization factor for the features and targets of a
        given HDF5 file.

        The normalization of the features is done through the NormParam class that assumes gaussian distribution.
        Hence the Normalized data should be normally distributed with a 0 mean value and 1 standard deviation.
        The normalization of the targets is done vian a min/max normalization. As a result the normalized targets
        should all lie between 0 and 1. By default the output file containing the normalization dictionary is called <hdf5name>_norm.pckl

        Args:

            fname (str): name of the hdf5 file
            shape (tuple(int), optional): shape of the grid in the hdf5 file

        Example:

            >>> norm = NormalizeData('1ak4.hdf5')
            >>> norm.get()
        """
        self.fname = fname
        self.parameters = {'features': {}, 'targets': {}}
        self.shape = shape
        self.fexport = os.path.splitext(self.fname)[0] + '_norm.pckl'
        self.skip_feature = []
        self.skip_target = []

    def get(self):
        """Get the normalization and write them to file."""

        self._extract_shape()
        self._load()
        self._extract_data()
        self._process_data()
        self._export_data()

    def _load(self):
        """Load data from already existing normalization file."""

        if os.path.isfile(self.fexport):

            f = open(self.fexport, 'rb')
            self.parameters = pickle.load(f)
            f.close()

            for _, feat_name in self.parameters['features'].items():
                for name, _ in feat_name.items():
                    self.skip_feature.append(name)

            for target in self.parameters['targets'].keys():
                self.skip_target.append(target)

    def _extract_shape(self):
        """Get the shape of the data in the hdf5 file."""

        if self.shape is not None:
            return

        f5 = h5py.File(self.fname, 'r')
        mol = list(f5.keys())[0]
        mol_data = f5.get(mol)

        if 'grid_points' in mol_data:

            nx = mol_data['grid_points']['x'].shape[0]
            ny = mol_data['grid_points']['y'].shape[0]
            nz = mol_data['grid_points']['z'].shape[0]
            self.shape = (nx, ny, nz)

        else:
            raise ValueError(
                'Impossible to determine sparse grid shape.\\n Specify argument grid_shape=(x,y,z)')

    def _extract_data(self):
        """Extract the data from the different maps."""

        f5 = h5py.File(self.fname, 'r')
        mol_names = list(f5.keys())
        self.nmol = len(mol_names)

        # loop over the molecules
        for mol in mol_names:

            # get the mapped features group
            data_group = f5.get(mol + '/mapped_features/')

            # loop over all the feature types
            for feat_types, feat_names in data_group.items():

                # if feature type not in param add
                if feat_types not in self.parameters['features']:
                    self.parameters['features'][feat_types] = {}

                # loop over all the feature
                for name in feat_names:

                    # we skip the target
                    if name in self.skip_feature:
                        continue

                    # create the param if it doesn't already exists
                    if name not in self.parameters['features'][feat_types]:
                        self.parameters['features'][feat_types][name] = NormParam(
                        )

                    # load the matrix
                    feat_data = data_group[feat_types + '/' + name]
                    if feat_data.attrs['sparse']:
                        mat = sparse.FLANgrid(sparse=True,
                                              index=feat_data['index'][:],
                                              value=feat_data['value'][:],
                                              shape=self.shape).to_dense()
                    else:
                        mat = feat_data['value'][:]

                    # add the parameter (mean and var)
                    self.parameters['features'][feat_types][name].add(
                        np.mean(mat), np.var(mat))

            # get the target groups
            target_group = f5.get(mol + '/targets')

            # loop over all the targets
            for tname, tval in target_group.items():

                # we skip the already computed target
                if tname in self.skip_target:
                    continue

                # create a new item if needed
                if tname not in self.parameters['targets']:
                    self.parameters['targets'][tname] = MinMaxParam()

                # update the value
                self.parameters['targets'][tname].update(tval[()])

        f5.close()

    def _process_data(self):
        """Compute the standard deviation of the data."""
        for feat_types, feat_dict in self.parameters['features'].items():
            for feat in feat_dict:
                self.parameters['features'][feat_types][feat].process(
                    self.nmol)

    def _export_data(self):
        """Pickle the data to file."""

        f = open(self.fexport, 'wb')
        pickle.dump(self.parameters, f)
        f.close()


class NormParam(object):

    def __init__(self, std=0, mean=0, var=0, sqmean=0):
        """Compute gaussian normalization for a given feature.

        This class allows to extract the standard deviation, mean value, variance and square root of the
        mean value of a mapped feature stored in the hdf5 file. As the entire data set is too large to fit in memory,
        the standard deviation of a given feature is calculated from the std of all the individual grids. This is done following:
        https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation:

        .. math::
            \\sigma_{tot}=\\sqrt{\\frac{1}{N}\\sum_i \\sigma_i^2+\\frac{1}{N}\\sum_i\\mu_i^2-(\\frac{1}{N}\\sum_i\\mu_i)^2}

        Args:
            std (float, optional): standard deviation
            mean (float,optional): mean value
            var (float,optional): variance
            sqmean (float, optional): square roo of the variance
        """

        self.std = std
        self.mean = mean
        self.var = var
        self.sqmean = sqmean

    def add(self, mean, var):
        """Add the mean value, sqmean and variance of a new molecule to the
        corresponding attributes."""
        self.mean += mean
        self.sqmean += mean**2
        self.var += var

    def process(self, n):
        """Compute the standard deviation of the ensemble."""

        # normalize the mean and var
        self.mean /= n
        self.var /= n
        self.sqmean /= n

        # get the std
        self.std = self.var
        self.std += self.sqmean
        self.std -= self.mean**2
        self.std = np.sqrt(self.std)


class MinMaxParam(object):

    """Compute the min/max of an ensenble of data.

    This is principally used to normalized the target values

    Args:
        minv (float, optional): minimal value
        maxv (float, optional): maximal value
    """

    def __init__(self, minv=None, maxv=None):
        self.min = minv
        self.max = maxv

    def update(self, val):

        if self.min is None:
            self.min = val
            self.max = val
        else:
            self.min = min(self.min, val)
            self.max = max(self.max, val)
