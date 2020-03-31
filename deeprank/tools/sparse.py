import numpy as np


def _printif(string, cond): return print(string) if cond else None


class FLANgrid(object):

    def __init__(self, sparse=None, index=None, value=None, shape=None):
        """Flat Array sparse matrix.

        Args:
            sparse (bool, optional): Sparse or Not
            index (list(int), optional): single index of each non-zero
                element
            value (list(float), optional): values of non-zero elements
            shape (3x3 array, optional): Shape of the matrix
        """
        self.sparse = sparse
        self.index = index
        self.value = value
        self.shape = shape

    def from_dense(self, data, beta=None, debug=False):
        """Create a sparse matrix from a dense one.

        Args:
            data (np.array): Dense matrix
            beta (float, optional): threshold to determine if a sparse
                rep is valuable
            debug (bool, optional): print debug information
        """
        if beta is not None:
            thr = beta * np.mean(np.abs(data))
            index = np.argwhere(np.abs(data) > thr)
            value = data[np.abs(data) > thr].reshape(-1, 1)
        else:
            index = np.argwhere(data != 0)
            value = data[data != 0].reshape(-1, 1)

        self.shape = data.shape

        # we can probably have different grid size
        # hence differnent index range to handle
        if np.prod(data.shape) < 2**16 - 1:
            index_type = np.uint16
            ind_byte = 16
        else:
            index_type = np.uint32
            ind_byte = 32

        # memory requirements
        mem_sparse = int(len(index) * ind_byte + len(index) * 32)
        mem_dense = int(np.prod(data.shape) * 32)

        # decide if we store sparse or not
        # if enough elements are sparse
        if mem_sparse < mem_dense:

            _printif(
                '--> FLAN sparse %d bits/%d bits' %
                (mem_sparse, mem_dense), debug)
            self.sparse = True
            self.index = self._get_single_index_array(index).astype(index_type)
            self.value = value.astype(np.float32)

        else:

            _printif(
                '--> FLAN dense %d bits/%d bits' %
                (mem_sparse, mem_dense), debug)
            self.sparse = False
            self.index = None
            self.value = data.astype(np.float32)

    def to_dense(self, shape=None):
        """Create a dense matrix.

        Args:
            shape (3x3 array, optional): Shape of the matrix

        Returns:
            np.array: Dense 3D matrix

        Raises:
            ValueError: shape not defined
        """
        if not self.sparse:
            raise ValueError('Not a sparse matrix')
        if shape is None and self.shape is None:
            raise ValueError('Shape not defined')
        if shape is None:
            shape = self.shape

        data = np.zeros(np.prod(self.shape))
        data[self.index] = self.value[:, 0]
        return data.reshape(self.shape)

    def _get_single_index(self, index):
        """Get the single index for a single element.

        # get the index can be used with a map
        # self.index = np.array( list( map(self._get_single_index,index) ) ).astype(index_type)
        # however that is remarkably slow compared to the array version

        Args:
            index (array): COO  index

        Returns:
            int: index
        """
        ndim = len(index)
        assert ndim == len(self.shape)

        ind = index[-1]
        for i in range(ndim - 1):
            ind += index[i] * np.prod(self.shape[i + 1:])
        return ind

    def _get_single_index_array(self, index):
        """Get the single index for multiple elements.

        # get the index can be used with a map
        # self.index = np.array( list( map(self._get_single_index,index) ) ).astype(index_type)
        # however that is remarkably slow compared to the array version

        Args:
            index (array): COO  index

        Returns:
            list(int): index
        """

        single_ind = index[:, -1]
        ndim = index.shape[-1]
        assert ndim == len(self.shape)

        for i in range(ndim - 1):
            single_ind += index[:, i] * np.prod(self.shape[i + 1:])

        return single_ind
