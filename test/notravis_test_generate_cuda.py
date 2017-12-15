import sys
import deeprank.generate

def test_generate(tune,test,gpu_block):

	# sources to assemble the data base
	pdb_source     = ['./1AK4/decoys/']
	pdb_native     = ['./1AK4/native/']

	#init the data assembler 
	database = deeprank.generate.DataGenerator(pdb_source=pdb_source,pdb_native=pdb_native,data_augmentation=None,
		                                       compute_targets  = ['deeprank.tools.targets.dockQ'],
		                                       compute_features = ['deeprank.tools.features.atomic'],
		                                       hdf5='./1ak4.hdf5',
	                                           )



	# map the features
	grid_info = {
		'atomic_densities' : {'CA':3.5,'N':3.5,'O':3.5,'C':3.5},
		'atomic_densities_mode' : 'sum',
		'number_of_points' : [30,30,30],
		'atomic_feature'  : ['vdwaals','coulomb','charge'],
		'atomic_feature_mode': 'sum',
		'resolution' : [1.,1.,1.]
	}

	# tune the kernel 
	if tune:
		database.tune_cuda_kernel(grid_info)
	elif test:
		database.test_cuda(grid_info,gpu_block)
	else:
		#create new files
		database.create_database()

		# map these data
		database.map_features(grid_info,cuda=True,gpu_block=gpu_block)


if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser(description='Map data using CUDA')
	parser.add_argument('--tune', action='store_true',help="tune the kernel")
	parser.add_argument('--test',action='store_true',help='test the kernel on 1 map')
	parser.add_argument('--gpu_block',nargs='+',default=[8,8,8],type=int,help='number of gpu block to use. Default: 8 8 1')
	args = parser.parse_args()

	test_generate(args.tune,args.test,args.gpu_block)


