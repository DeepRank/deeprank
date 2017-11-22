import os
import numpy as np

class FeatureClass(object):

	'''
	Master class fron which all the other classes
	should be derived


	self.feature_data : dictionary of features
	                    {'coulomb':data_dict_clb,
	                     'vdwaals':data_dict_vdw }

	The data_dict_xxx must be a dictioanry with

	'''

	def __init__(self,feature_type):

		self.type = feature_type
		self.feature_data = {}	
		self.export_directories = {}	

	# eport the data 
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



	# export hdf5 as line text
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

