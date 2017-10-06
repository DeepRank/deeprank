import numpy 
import subprocess as sp
import pickle 
import sys 
import numpy as np

def merge_database(new_db = None, root_db = None ):


	if new_db is None or root_db is None:
		print('You must give  new and root database')
		sys.exit()

	new_folders = sp.check_output('ls %s' %new_db,shell=True).decode('utf-8').split()
	root_folders = sp.check_output('ls %s' %root_db,shell=True).decode('utf-8').split()

	# extract the molecule names in the db
	new_mol_names = [ mol.split('/')[-1] for mol in new_folders]
	root_mol_names = [ mol.split('/')[-1] for mol in root_folders]


	# loop over the new folders
	for sub, mol_name in zip(new_folders,new_mol_names):

		# check similar folder in root
		if mol_name in root_mol_names:

			print('Merging %s in root folder %s' %(mol_name,root_db))

			# examine what's inside
			data_dir = sp.check_output('ls %s/%s' %(new_db,sub),shell=True).decode('utf-8').split()
			data_dir = [s.split('/')[-1] for s in data_dir]

			if 'input' in data_dir:

				# load the grid point
				grid_new = np.load('%s/%s/input/grid_points.npz' %(new_db,mol_name))
				grid_root = np.load('%s/%s/input/grid_points.npz' %(root_db,mol_name))

				# check if they are the same
				dx = np.abs(np.sum(grid_new['x'] - grid_root['x']))
				dy = np.abs(np.sum(grid_new['y'] - grid_root['y']))
				dz = np.abs(np.sum(grid_new['z'] - grid_root['z']))
				dtot = dx+dy+dz
				
				if dtot > 1E-6:
					print('Error grids incompatible. Cannot merge database')
					sys.exit()

				# get all the pkl file
				new_pkl = sp.check_output('ls %s/%s/input/*.pkl' %(new_db,mol_name),shell=True).decode('utf-8').split()
				root_pkl = sp.check_output('ls %s/%s/input/*.pkl' %(root_db,mol_name),shell=True).decode('utf-8').split()
				new_pkl_names = [f.split('/')[-1] for f in new_pkl]
				root_pkl_names = [f.split('/')[-1] for f in root_pkl]

				for file,file_name in zip(new_pkl,new_pkl_names):

					if file_name in root_pkl_names:
						new_data = pickle.load(open(file,'rb'))
						root_file = '%s/%s/input/%s' %(root_db,mol_name,file_name)
						root_data = pickle.load(open(root_file,'rb'))

						# create the new dict
						new_data = {**root_data,**new_data}

						# pickle it
						pickle.dump(new_data,open(root_file,'wb'))


			if 'targets' in data_dir:

				sp.call('cp %s/%s/targets/* %s/%s/targets/' %(new_db,mol_name,root_db,mol_name))


		# it it's an entirely new molecule
		else:
			sp.call('cp %s/%s %s' %(new_db,mol_name,root_db))

if __name__ == "__main__":

	merge_database(new_db = sys.argv[1],root_db = sys.argv[2])