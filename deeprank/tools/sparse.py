import numpy as np


'''
Routines to handle sparse tensors
'''

class COOgrid(object):

	def __init__(self,sparse=None,index=None,value=None,shape=None):

		self.sparse=sparse
		self.index = index
		self.value = value
		self.shape = shape

	def from_dense(self,data,beta=None):
		'''
		data : the 3D tensor to encode
		beta : threshold to determine if a sparse rep is valuable 
		'''
		if beta is not None:
			thr = beta*np.mean(np.abs(data))
			index = np.argwhere(np.abs(data)>thr)
			value = data[np.abs(data)>thr].reshape(-1,1)
		else:
			index = np.argwhere(data!=0)
			value = data[data!=0].reshape(-1,1)

		# memory requirements
		mem_sparse = int(len(index)*(data.ndim)*8 + len(index) * 32)
		mem_dense = int(np.prod(data.shape)*32)

		# decide if we store sparse or not
		# if enough elements are sparse
		if mem_sparse < mem_dense:		
			print('--> COO sparse %d bits/%d bits' %(mem_sparse,mem_dense))
			self.sparse = True
			self.index = index.astype(np.uint8)
			self.value= value.astype(np.float32)		
			self.shape = data.shape

		else:
			print('--> dense %d bits/%d bits' %(mem_sparse,mem_dense))
			self.sparse = False
			self.shape = data.shape
			self.index=None
			self.value=data.astype(np.float32)	

	def to_dense(self,shape=None):
		if not self.sparse:
			raise ValueError('Not a sparse matrix')
		if shape is None and self.shape is None:
			raise ValueError('Shape not defined')
		if shape is None:
			shape = self.shape

		data = np.zeros(shape)
		data[[ i for i in self.index.T ]] = self.value[:,0]
		return data

'''
Flat Array Normalized 
'''
class FLANgrid(object):

	def __init__(self,sparse=None,index=None,value=None,shape=None):

		self.sparse=sparse
		self.index = index
		self.value = value
		self.shape = shape

	def from_dense(self,data,beta=None):
		'''
		data : the 3D tensor to encode
		beta : threshold to determine if a sparse rep is valuable 
		'''
		if beta is not None:
			thr = beta*np.mean(np.abs(data))
			index = np.argwhere(np.abs(data)>thr)
			value = data[np.abs(data)>thr].reshape(-1,1)
		else:
			index = np.argwhere(data!=0)
			value = data[data!=0].reshape(-1,1)

		self.shape = data.shape

		# we can probably have different grid size
		# hence differnent index range to handle
		if np.prod(data.shape) < 2**16-1:
			index_type = np.uint16
			ind_byte = 16
		else:
			index_type = np.uint32
			ind_byte = 32

		# memory requirements
		mem_sparse = int(len(index)*ind_byte + len(index) * 32)
		mem_dense = int(np.prod(data.shape)*32)

		# decide if we store sparse or not
		# if enough elements are sparse
		if mem_sparse < mem_dense:		
			#print('--> FLAN sparse %d bits/%d bits' %(mem_sparse,mem_dense))
			self.sparse = True
			self.index = np.array( list( map(self._get_single_index,index) ) ).astype(index_type)
			self.value= value.astype(np.float32)		
			

		else:
			#print('--> FLAN dense %d bits/%d bits' %(mem_sparse,mem_dense))
			self.sparse = False
			self.index=None
			self.value=data.astype(np.float32)	

	def to_dense(self,shape=None):
		if not self.sparse:
			raise ValueError('Not a sparse matrix')
		if shape is None and self.shape is None:
			raise ValueError('Shape not defined')
		if shape is None:
			shape = self.shape

		data = np.zeros(np.prod(self.shape))
		data[self.index] = self.value[:,0]
		return data.reshape(self.shape)

	# get the index
	def _get_single_index(self,index):
		
		ndim = len(index)
		assert ndim == len(self.shape)

		ind = index[-1]
		for i in range(ndim-1):
			ind += index[i] * np.prod(self.shape[i+1:])
		return ind
