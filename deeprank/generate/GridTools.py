
import itertools
import sys
from time import time

import numpy as np
from scipy.signal import bspline
import pdb2sql

from deeprank.config import logger
from deeprank.tools import sparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x


def logif(string, cond): return logger.info(string) if cond else None


class GridTools(object):

    def __init__(self, molgrp, chain1, chain2,
                 number_of_points=30, resolution=1.,
                 atomic_densities=None, atomic_densities_mode='ind',
                 feature=None, feature_mode='ind',
                 contact_distance=8.5,
                 cuda=False, gpu_block=None, cuda_func=None, cuda_atomic=None,
                 prog_bar=False, time=False, try_sparse=True):
        """Map the feature of a complex on the grid.

        Args:
            molgrp(str): name of the group of the molecule in the HDF5 file.
            chain1 (str): First chain ID.
            chain2 (str): Second chain ID.
            number_of_points(int, optional): number of points we want in
                each direction of the grid.
            resolution(float, optional): distance(in Angs) between two points.
            atomic_densities(dict, optional): dictionary of element types with
                their vdw radius, see deeprank.config.atom_vdw_radius_noH
            atomic_densities_mode(str, optional): Mode for mapping
                (deprecated must be 'ind').
            feature(None, optional): Name of the features to be mapped.
                By default all the features present in
                hdf5_file['< molgrp > /features/] will be mapped.
            feature_mode(str, optional): Mode for mapping
                (deprecated must be 'ind').
            contact_distance(float, optional): the dmaximum distance
                between two contact atoms default 8.5Å.
            cuda(bool, optional): Use CUDA or not.
            gpu_block(tuple(int), optional): GPU block size to use.
            cuda_func(None, optional): Name of the CUDA function to be
                used for the mapping of the features.
                Must be present in kernel_cuda.c.
            cuda_atomic(None, optional): Name of the CUDA function to be
                used for the mapping of the atomic densities.
                Must be present in kernel_cuda.c.
            prog_bar(bool, optional): print progression bar for
                individual grid (default False).
            time(bool, optional): print timing statistic for
                individual grid (default False).
            try_sparse(bool, optional): Try to store the matrix in
                sparse format (default True).
        """

        # mol and hdf5 file
        self.molgrp = molgrp
        self.mol_basename = molgrp.name

        # chain IDs
        self.chain1 = chain1
        self.chain2 = chain2

        # hdf5 file to strore data
        self.hdf5 = self.molgrp.file
        self.try_sparse = try_sparse

        # parameter of the grid
        if number_of_points is not None:
            if not isinstance(number_of_points, list):
                number_of_points = [number_of_points] * 3
            self.npts = np.array(number_of_points).astype('int')

        if resolution is not None:
            if not isinstance(resolution, list):
                resolution = [resolution] * 3
            self.res = np.array(resolution)

        # feature requested
        self.atomic_densities = atomic_densities
        self.feature = feature

        # mapping mode
        self.feature_mode = feature_mode
        self.atomic_densities_mode = atomic_densities_mode

        # cuda support
        self.cuda = cuda
        if self.cuda:  # pragma: no cover
            self.gpu_block = gpu_block
            self.gpu_grid = [int(np.ceil(n / b))
                             for b, n in zip(self.gpu_block, self.npts)]

        # cuda
        self.cuda_func = cuda_func
        self.cuda_atomic = cuda_atomic

        # parameter of the atomic system
        self.atom_xyz = None
        self.atom_index = None
        self.atom_type = None

        # grid points
        self.x = None
        self.y = None
        self.z = None

        # grids for calculation of atomic densities
        self.xgrid = None
        self.ygrid = None
        self.zgrid = None

        # dictionaries of atomic densities
        self.atdens = {}

        # conversion from boh to angs for VMD visualization
        self.bohr2ang = 0.52918

        # contact distance to locate the interface
        self.contact_distance = contact_distance

        # progress bar
        self.local_tqdm = lambda x: tqdm(x) if prog_bar else x
        self.time = time

        # if we already have an output containing the grid
        # we update the existing features
        _update_ = False
        if self.mol_basename + '/grid_points/x' in self.hdf5:
            _update_ = True

        if _update_:
            logif(f'\n=Updating grid data for {self.mol_basename}.',
                  self.time)
            self.update_feature()
        else:
            logif(f'\n= Creating grid and grid data for {self.mol_basename}.',
                  self.time)
            self.create_new_data()

    ################################################################

    def create_new_data(self):
        """Create new feature for a given complex."""

        # get the position/atom type .. of the complex
        self.read_pdb()

        # get the contact atoms and interface center
        self.get_contact_center()

        # define the grid
        self.define_grid_points()

        # save the grid points
        self.export_grid_points()

        # map the features
        self.add_all_features()

        # if we wnat the atomic densisties
        self.add_all_atomic_densities()

        # cloe the db file
        self.sqldb._close()

    ################################################################

    def update_feature(self):
        """Update existing feature in a complex."""

        # get the position/atom type .. of the complex
        # get self.sqldb
        self.read_pdb()

        # read the grid from the hdf5
        grid = self.hdf5.get(self.mol_basename + '/grid_points/')
        self.x, self.y, self.z = grid['x'][()], grid['y'][()], grid['z'][()]

        # create the grid
        self.ygrid, self.xgrid, self.zgrid = np.meshgrid(
            self.y, self.x, self.z)

        # set the resolution/dimension
        self.npts = np.array([len(self.x), len(self.y), len(self.z)])
        self.res = np.array(
            [self.x[1] - self.x[0], self.y[1] - self.y[0], self.z[1] - self.z[0]])

        # map the features
        self.add_all_features()

        # if we want the atomic densisties
        self.add_all_atomic_densities()

        # cloe the db file
        self.sqldb._close()

    ################################################################

    def read_pdb(self):
        """Create a sql databse for the pdb."""

        self.sqldb = pdb2sql.interface(self.molgrp['complex'][()])

    # get the contact atoms and interface center
    def get_contact_center(self):
        """Get the center of conact atoms."""

        contact_atoms = self.sqldb.get_contact_atoms(
            cutoff=self.contact_distance, chain1=self.chain1, chain2=self.chain2)

        tmp = []
        for i in contact_atoms.values():
            tmp.extend(i)
        contact_atoms = list(set(tmp))

        # get interface center
        self.center_contact = np.mean(
            np.array(self.sqldb.get('x,y,z', rowID=contact_atoms)), 0)

    ################################################################
    # shortcut to add all the feature a
    # and atomic densities in just one line
    ################################################################

    # add all the residue features to the data

    def add_all_features(self):
        """Add all the features toa given molecule."""

        # map the features
        if self.feature is not None:

            # map the residue features
            dict_data = self.map_features(self.feature)

            # save to hdf5 if specfied
            t0 = time()
            logif('-- Save Features to HDF5', self.time)
            self.hdf5_grid_data(dict_data, 'Feature_%s' % (self.feature_mode))
            logif('      Total %f ms' % ((time() - t0) * 1000), self.time)

    # add all the atomic densities to the data

    def add_all_atomic_densities(self):
        """Add all atomic densities."""

        # if we wnat the atomic densisties
        if self.atomic_densities is not None:

            # compute the atomic densities
            self.map_atomic_densities()

            # save to hdf5
            t0 = time()
            logif('-- Save Atomic Densities to HDF5', self.time)
            self.hdf5_grid_data(self.atdens, 'AtomicDensities_%s' %
                                (self.atomic_densities_mode))
            logif('      Total %f ms' % ((time() - t0) * 1000), self.time)

    ################################################################
    # define the grid points
    # there is an issue maybe with the ordering
    # In order to visualize the data in VMD the Y and X axis must be inverted ...
    # I keep it like that for now as it should not matter for the CNN
    # and maybe we don't need atomic denisties as features
    ################################################################

    def define_grid_points(self):
        """Define the grid points."""

        logif('-- Define %dx%dx%d grid ' %
              (self.npts[0], self.npts[1], self.npts[2]), self.time)
        logif('-- Resolution of %1.2fx%1.2fx%1.2f Angs' %
              (self.res[0], self.res[1], self.res[2]), self.time)

        halfdim = 0.5 * (self.npts * self.res)
        center = self.center_contact

        low_lim = center - halfdim
        hgh_lim = low_lim + self.res * (np.array(self.npts) - 1)

        self.x = np.linspace(low_lim[0], hgh_lim[0], self.npts[0])
        self.y = np.linspace(low_lim[1], hgh_lim[1], self.npts[1])
        self.z = np.linspace(low_lim[2], hgh_lim[2], self.npts[2])

        # there is something fishy about the meshgrid 3d
        # the axis are a bit screwy ....
        # i dont quite get why the ordering is like that
        self.ygrid, self.xgrid, self.zgrid = np.meshgrid(
            self.y, self.x, self.z)

    ################################################################
    # Atomic densities
    # as defined in the paper about ligand in protein
    ################################################################

    # compute all the atomic densities data
    def map_atomic_densities(self, only_contact=True):
        """Map the atomic densities to the grid.

        Args:
            only_contact(bool, optional): Map only the contact atoms

        Raises:
            ImportError: Description
        """
        mode = self.atomic_densities_mode
        logif('-- Map atomic densities on %dx%dx%d grid (mode=%s)' %
              (self.npts[0], self.npts[1], self.npts[2], mode), self.time)

        # prepare the cuda memory
        if self.cuda:  # pragma: no cover

            # try to import pycuda
            try:
                from pycuda import driver, compiler, gpuarray, tools
                import pycuda.autoinit
            except BaseException:
                raise ImportError("Error when importing pyCuda in GridTools")

            # book mem on the gpu
            x_gpu = gpuarray.to_gpu(self.x.astype(np.float32))
            y_gpu = gpuarray.to_gpu(self.y.astype(np.float32))
            z_gpu = gpuarray.to_gpu(self.z.astype(np.float32))
            grid_gpu = gpuarray.zeros(self.npts, np.float32)

        # get the contact atoms
        if only_contact:
            index = self.sqldb.get_contact_atoms(cutoff=self.contact_distance,
                chain1=self.chain1, chain2=self.chain2)
        else:
            index = {self.chain1: self.sqldb.get('rowID', chainID=self.chain1),
                     self.chain2: self.sqldb.get('rowID', chainID=self.chain2)}

        # loop over all the data we want
        for elementtype, vdw_rad in self.local_tqdm(
                self.atomic_densities.items()):

            t0 = time()

            xyzA = np.array(self.sqldb.get(
                'x,y,z', rowID=index[self.chain1], element=elementtype))
            xyzB = np.array(self.sqldb.get(
                'x,y,z', rowID=index[self.chain2], element=elementtype))

            tprocess = time() - t0

            t0 = time()
            # if we use CUDA
            if self.cuda:  # pragma: no cover

                # reset the grid
                grid_gpu *= 0

                # get the atomic densities of chain A
                for pos in xyzA:
                    x0, y0, z0 = pos.astype(np.float32)
                    vdw = np.float32(vdw_rad)
                    self.cuda_atomic(
                        vdw, x0, y0, z0, x_gpu, y_gpu, z_gpu, grid_gpu, block=tuple(
                            self.gpu_block), grid=tuple(
                            self.gpu_grid))
                    atdensA = grid_gpu.get()

                # reset the grid
                grid_gpu *= 0

                # get the atomic densities of chain B
                for pos in xyzB:
                    x0, y0, z0 = pos.astype(np.float32)
                    vdw = np.float32(vdw_rad)
                    self.cuda_atomic(
                        vdw, x0, y0, z0, x_gpu, y_gpu, z_gpu, grid_gpu, block=tuple(
                            self.gpu_block), grid=tuple(
                            self.gpu_grid))
                    atdensB = grid_gpu.get()

            # if we don't use CUDA
            else:

                # init the grid
                atdensA = np.zeros(self.npts)
                atdensB = np.zeros(self.npts)

                # run on the atoms
                for pos in xyzA:
                    atdensA += self.densgrid(pos, vdw_rad)

                # run on the atoms
                for pos in xyzB:
                    atdensB += self.densgrid(pos, vdw_rad)

            # create the final grid: A - B
            if mode == 'diff':
                self.atdens[elementtype] = atdensA - atdensB

            # create the final grid: A + B
            elif mode == 'sum':
                self.atdens[elementtype] = atdensA + atdensB

            # create the final grid: A and B
            elif mode == 'ind':
                self.atdens[elementtype + '_chain1'] = atdensA
                self.atdens[elementtype + '_chain2'] = atdensB
            else:
                raise ValueError(f'Atomic density mode {mode} not recognized')

            tgrid = time() - t0
            logif('     Process time %f ms' % (tprocess * 1000), self.time)
            logif('     Grid    time %f ms' % (tgrid * 1000), self.time)

    # compute the atomic denisties on the grid
    def densgrid(self, center, vdw_radius):
        """Function to map individual atomic density on the grid.

        The formula is equation (1) of the Koes paper
        Protein-Ligand Scoring with Convolutional NN Arxiv:1612.02751v1

        Args:
            center (list(float)): position of the atoms
            vdw_radius (float): vdw radius of the atom

        Returns:
            TYPE: np.array (mapped density)
        """

        x0, y0, z0 = center
        dd = np.sqrt((self.xgrid - x0)**2
                     + (self.ygrid - y0)**2
                     + (self.zgrid - z0)**2)

        dgrid = np.zeros(self.npts)

        index_shortd = dd < vdw_radius
        index_longd = (dd >= vdw_radius) & (dd < 1.5 * vdw_radius)
        dgrid[index_shortd] = np.exp(-2 * dd[index_shortd]**2 / vdw_radius**2)
        dgrid[index_longd] = 4. / np.e**2 / vdw_radius**2 * dd[index_longd]**2 \
            - 12. / np.e**2 / vdw_radius * dd[index_longd] + 9. / np.e**2
        return dgrid

    ################################################################
    # Residue or Atomic features
    # read the file provided in input
    # and map it on the grid
    ################################################################

    # map residue a feature on the grid
    def map_features(self, featlist, transform=None):
        """Map individual feature to the grid.

        For residue based feature the feature file must be of the format
        chainID residue_name(3-letter)  residue_number [values]

        For atom based feature it must be
        chainID residue_name(3-letter)  residue_number atome_name [values]

        Args:
            featlist (list(str)): list of features to be mapped
            transform (callable, optional): transformation of the feature (?)

        Returns:
            np.array: Mapped features

        Raises:
            ImportError: Description
            ValueError: Description
        """

        # declare the total dictionary
        dict_data = {}

        # prepare the cuda memory
        if self.cuda:  # pragma: no cover

            # try to import pycuda
            try:
                from pycuda import driver, compiler, gpuarray, tools
                import pycuda.autoinit
            except BaseException:
                raise ImportError("Error when importing pyCuda in GridTools")

            # book mem on the gpu
            x_gpu = gpuarray.to_gpu(self.x.astype(np.float32))
            y_gpu = gpuarray.to_gpu(self.y.astype(np.float32))
            z_gpu = gpuarray.to_gpu(self.z.astype(np.float32))
            grid_gpu = gpuarray.zeros(self.npts, np.float32)

        # loop over all the features required
        for feature_name in featlist:

            logif('-- Map %s on %dx%dx%d grid '
                  % (feature_name, self.npts[0],
                     self.npts[1], self.npts[2]), self.time)

            # read the data
            featgrp = self.molgrp['features']
            if feature_name in featgrp.keys():
                data = featgrp[feature_name][:]
            else:
                print('Error Feature not found \n\tPossible features: ' +
                      ' | '.join(featgrp.keys()))
                raise ValueError(
                    'feature %s  not found in the file' % (feature_name))

            # detect if we have a xyz format
            # or a byte format
            # define how many elements (ntext)
            # are present before the feature values
            # xyz: 4 (chain x y z)
            # byte - residue: 3 (chain resSeq resName)
            # byte - atomic: 4 (chain resSeq resName name)

            # all the format are now xyz
            feature_type = 'xyz'
            ntext = 4

            # test if the transform is callable
            # and test it on the first line of the data
            # get the data on the first line
            if data.shape[0] != 0:

                data_test = data[0, ntext:]

                # define the length of the output
                if transform is None:
                    nFeat = len(data_test)
                elif callable(transform):
                    nFeat = len(transform(data_test))
                else:
                    print('Error transform in map_feature must be callable')
                    return None
            else:
                nFeat = 1

            # declare the dict
            # that will in fine holds all the data
            if nFeat == 1:
                if self.feature_mode == 'ind':
                    dict_data[feature_name + '_chain1'] = np.zeros(self.npts)
                    dict_data[feature_name + '_chain2'] = np.zeros(self.npts)
                else:
                    dict_data[feature_name] = np.zeros(self.npts)
            else: # do we need that ?!
                for iF in range(nFeat):
                    if self.feature_mode == 'ind':
                        dict_data[feature_name + '_chain1_%03d' %
                                  iF] = np.zeros(self.npts)
                        dict_data[feature_name + '_chain2_%03d' %
                                  iF] = np.zeros(self.npts)
                    else:
                        dict_data[feature_name + '_%03d' %
                                  iF] = np.zeros(self.npts)

            # skip empty features
            if data.shape[0] == 0:
                continue

            # rest the grid and get the x y z values
            if self.cuda:  # pragma: no cover
                grid_gpu *= 0

            # timing
            tprocess = 0
            tgrid = 0

            # map all the features
            for line in self.local_tqdm(data):
                t0 = time()
                # if the feature was written with xyz data
                # i.e chain x y z values
                if feature_type == 'xyz':

                    chain = [self.chain1, self.chain2][int(line[0])]
                    pos = line[1:ntext]
                    feat_values = np.array(line[ntext:])

                # if the feature was written with bytes
                # i.e chain resSeq resName (name) values
                # TODO deprecated?
                else:

                    # decode the line
                    line = line.decode('utf-8').split()

                    # get the position of the resnumber
                    chain, resName, resNum = line[0], line[1], line[2]

                    # get the atom name for atomic data
                    if feature_type == 'atomic':
                        atName = line[3]

                    # get the position
                    # TODO deprecated? the definition of center postion is
                    #  different from that in features e.g. BSA.py
                    if feature_type == 'residue':
                        pos = np.mean(np.array(self.sqldb.get(
                            'x,y,z', chainID=chain, resSeq=resNum)), 0)
                        sql_resName = list(set(self.sqldb.get(
                            'resName', chainID=chain, resSeq=resNum)))
                    else:
                        pos = np.array(self.sqldb.get(
                            'x,y,z', chainID=chain,
                            resSeq=resNum, name=atName))[0]
                        sql_resName = list(set(self.sqldb.get(
                            'resName', chainID=chain,
                            resSeq=resNum, name=atName)))

                    # check if  the resname correspond
                    if len(sql_resName) == 0:
                        print('Error: SQL query returned empty list')
                        print('Tip  : Make sure the parameter file ')
                        print('Tip  : corresponds to the pdb file %s' %
                              (self.sqldb.pdbfile))
                        sys.exit()
                    else:
                        sql_resName = sql_resName[0]

                    if resName != sql_resName:
                        print('Residue Name Error in the Feature file ')
                        print('Feature File: chain %s resNum %s  resName %s' %
                              (chain, resNum, resName))
                        print('SQL data    : chain %s resNum %s  resName %s' %
                              (chain, resNum, sql_resName))
                        sys.exit()

                    # get the values of the feature(s) for thsi residue
                    feat_values = np.array(list(map(float, line[ntext:])))

                # postporcess the data
                if callable(transform):
                    feat_values = transform(feat_values)

                # handle the mode
                fname = feature_name
                if self.feature_mode == "diff":
                    coeff = {self.chain1: 1, self.chain2: -1}[chain]
                else:
                    coeff = 1
                if self.feature_mode == "ind":
                    chain_name = {self.chain1: '1', self.chain2: '2'}[chain]
                    fname = feature_name + "_chain" + chain_name
                tprocess += time() - t0

                t0 = time()
                # map this feature(s) on the grid(s)
                if not self.cuda:
                    if nFeat == 1:
                        dict_data[fname] += coeff * \
                            self.featgrid(pos, feat_values)
                    else:
                        for iF in range(nFeat):
                            dict_data[fname + '_%03d' % iF] += coeff * \
                                self.featgrid(pos, feat_values[iF])

                # try to use cuda to speed it up
                else:  # pragma: no cover
                    if nFeat == 1:
                        x0, y0, z0 = pos.astype(np.float32)
                        alpha = np.float32(coeff * feat_values)
                        self.cuda_func(alpha,
                                       x0, y0, z0,
                                       x_gpu, y_gpu, z_gpu,
                                       grid_gpu,
                                       block=tuple(self.gpu_block),
                                       grid=tuple(self.gpu_grid))
                    else:
                        raise ValueError(
                            'CUDA only possible for single-valued features')

                tgrid += time() - t0

            if self.cuda:  # pragma: no cover
                dict_data[fname] = grid_gpu.get()
                driver.Context.synchronize()

            logif('     Process time %f ms' % (tprocess * 1000), self.time)
            logif('     Grid    time %f ms' % (tgrid * 1000), self.time)

        return dict_data

    # compute the a given feature on the grid
    def featgrid(self, center, value, type_='fast_gaussian'):
        """Map an individual feature (atomic or residue) on the grid.

        Args:
            center (list(float)): position of the feature center
            value (float): value of the feature
            type_ (str, optional): method to map

        Returns:
            np.array: Mapped feature

        Raises:
            ValueError: Description
        """

        # shortcut for th center
        x0, y0, z0 = center
        sigma = np.sqrt(1. / 2)
        beta = 0.5 / (sigma**2)

        # simple Gaussian
        if type_ == 'gaussian':
            dd = np.sqrt((self.xgrid - x0)**2
                         + (self.ygrid - y0)**2
                         + (self.zgrid - z0)**2)
            dd = value * np.exp(-beta * dd)
            return dd

        # fast gaussian
        elif type_ == 'fast_gaussian':

            cutoff = 5. * beta

            dd = np.sqrt((self.xgrid - x0)**2
                         + (self.ygrid - y0)**2
                         + (self.zgrid - z0)**2)
            dgrid = np.zeros(self.npts)

            dgrid[dd < cutoff] = value * np.exp(-beta * dd[dd < cutoff])

            return dgrid

        # Bsline
        elif type_ == 'bspline':
            spline_order = 4
            spl = bspline((self.xgrid - x0) / self.res[0], spline_order) \
                * bspline((self.ygrid - y0) / self.res[1], spline_order) \
                * bspline((self.zgrid - z0) / self.res[2], spline_order)
            dd = value * spl
            return dd

        # nearest neighbours
        elif type_ == 'nearest':

            # distances
            dx = np.abs(self.x - x0)
            dy = np.abs(self.y - y0)
            dz = np.abs(self.z - z0)

            # index
            indx = np.argsort(dx)[:2]
            indy = np.argsort(dy)[:2]
            indz = np.argsort(dz)[:2]

            # weight
            wx = dx[indx]
            wx /= np.sum(wx)

            wy = dy[indy]
            wy /= np.sum(wy)

            wz = dx[indz]
            wz /= np.sum(wz)

            # define the points
            indexes = [indx, indy, indz]
            points = list(itertools.product(*indexes))

            # define the weight
            W = [wx, wy, wz]
            W = list(itertools.product(*W))
            W = [np.sum(iw) for iw in W]

            # put that on the grid
            dgrid = np.zeros(self.npts)

            for w, pt in zip(W, points):
                dgrid[pt[0], pt[1], pt[2]] = w * value

            return dgrid

        # default
        else:
            raise ValueError(f'Options not recognized for the grid {type_}')

    ################################################################
    # export the grid points for external calculations of some
    # features. For example the electrostatic potential etc ...
    ################################################################

    def export_grid_points(self):
        """export the grid points to the hdf5 file."""

        grd = self.hdf5.require_group(self.mol_basename + '/grid_points')
        grd.create_dataset('x', data=self.x)
        grd.create_dataset('y', data=self.y)
        grd.create_dataset('z', data=self.z)

        # add center or update it when the old value is different
        if 'center' not in grd:
            grd.create_dataset('center', data=self.center_contact)
        elif not all(grd['center'][()] == self.center_contact):
            grd['center'][...] = self.center_contact

    # save the data in the hdf5 file

    def hdf5_grid_data(self, dict_data, data_name):
        """Save the mapped feature to the hdf5 file.

        Args:
            dict_data(dict): feature values stored as a dict
            data_name(str): feature name
        """
        # get the group og the feature
        feat_group = self.hdf5.require_group(
            self.mol_basename + '/mapped_features/' + data_name)

        # gothrough all the feature elements
        for key, value in dict_data.items():

            # remove only subgroup
            if key in feat_group:
                del feat_group[key]

            # create new one
            sub_feat_group = feat_group.create_group(key)

            # try  a sparse representation
            if self.try_sparse:

                # check if the grid is sparse or not
                t0 = time()
                spg = sparse.FLANgrid()
                spg.from_dense(value, beta=1E-2)
                if self.time:
                    print('      Sparsing time %f ms' % ((time() - t0) * 1000))

                # if we have a sparse matrix
                if spg.sparse:
                    sub_feat_group.attrs['sparse'] = spg.sparse
                    sub_feat_group.attrs['type'] = 'sparse_matrix'
                    sub_feat_group.create_dataset(
                        'index', data=spg.index,
                        compression='gzip', compression_opts=9)
                    sub_feat_group.create_dataset(
                        'value', data=spg.value,
                        compression='gzip', compression_opts=9)

                else:
                    sub_feat_group.attrs['sparse'] = spg.sparse
                    sub_feat_group.attrs['type'] = 'dense_matrix'
                    sub_feat_group.create_dataset(
                        'value', data=spg.value,
                        compression='gzip', compression_opts=9)

            else:
                sub_feat_group.attrs['sparse'] = False
                sub_feat_group.attrs['type'] = 'dense_matrix'
                sub_feat_group.create_dataset(
                    'value', data=value,
                    compression='gzip', compression_opts=9)

########################################################################
