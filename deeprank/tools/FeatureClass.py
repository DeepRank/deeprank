import os
import numpy as np

class FeatureClass(object):

	'''
	Master class fron which all the other classes
	should be derived


	self.feature_data : dictionary of features
	                    {'coulomb':data_dict_clb[(atom info):value] 
	                     'vdwaals':data_dict_vdw[(atom info):value]  }

	self.feature_data_xyz : dictionary of feature
							dictionary of features
	                    {'coulomb':data_dict_clb[(atom xyz):value] 
	                     'vdwaals':data_dict_vdw[(atom xyz):value]  }

	'''

	def __init__(self,feature_type):

		self.type = feature_type
		self.feature_data = {}	
		self.feature_data_xyz = {}	
		self.export_directories = {}	

	########################################
	#
	# export the data in a singl file
	# Pretty sure we neve use that anymore
	# I jsut keep it for legacy reasons
	#
	########################################
	def export_data(self,mol_name):


		for name,data in self.feature_data.items():

			dirname = self.export_directories[name]

			if dirname[-1] != '/':
				dirname += '/'

			if not os.path.isdir(dirname):
				os.mkdir(dirname)

			filename = dirname + mol_name + '.' + name.upper()

			f = open(filename,'w')

			for key,value in data.items():

				# residue based feature
				if len(key) == 3:

					# tags
					feat = '{:>4}{:>10}{:>10}'.format(key[0],key[1],key[2])

				# atomic based features
				elif len(key) == 4:

					# tags
					feat = '{:>4}{:>10}{:>10}{:>10}'.format(key[0],key[1],key[2],key[3])

				# values
				for v in value:
					feat += '\t{: 1.6E}'.format(v)

				feat += '\n'
				f.write(feat)

			f.close()


	########################################
	#
	# export the data in an HDF5 file group
	# the format of the data is here
	# 
	# for atomic features
	# chainID  resSeq resNum name [values] 
	#
	# for residue features
	# chainID  resSeq resNum [values]
	#  
	# PRO : might be usefull for other applications
	# CON : slow when mapping cause we have to retrive the xyz
	#	
	########################################
	def export_data_hdf5(self,featgrp):

		# loop through the datadict and name
		for name,data in self.feature_data.items():	

			ds = []
			for key,value in data.items():

				# residue based feature
				if len(key) == 3:

					# tags
					feat = '{:>4}{:>10}{:>10}'.format(key[0],key[1],key[2])

				# atomic based features
				elif len(key) == 4:

					# tags
					feat = '{:>4}{:>10}{:>10}{:>10}'.format(key[0],key[1],key[2],key[3])

				# values
				for v in value:
					feat += '    {: 1.6E}'.format(v)	

				# append
				ds.append(feat)

			# put in the hdf5 file
			ds = np.array(ds).astype('|S'+str(len(ds[0])))

			# create the dataset
			featgrp.create_dataset(name,data=ds)			



	########################################
	#
	# export the data in an HDF5 file group
	# the format of the data is here
	# 
	# for atomic and residue features
	# x y z [values] 
	#  
	# PRO : fast when mapping 
	# CON : only usefull for deeprank
	#
	########################################
	def export_dataxyz_hdf5(self,featgrp):

		# loop through the datadict and name
		for name,data in self.feature_data_xyz.items():	

			# create the data set
			ds = np.array([list(key)+value for key,value in data.items()])

			# create the dataset
			featgrp.create_dataset(name,data=ds)	