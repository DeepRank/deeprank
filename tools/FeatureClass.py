
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

	# eport the data 
	def export_data(self):


		for name,data in self.feature_data.items():

			filename = name+'.dat'

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