import deeprank.map

##########################################################################
#
# Use the GridTool class to map the features of one complex only
# and create the cube files and VMD scripts to visulize the data
#
##########################################################################


grid = deeprank.map.GridTools(mol_name='./gridtest/complex.pdb',
	             number_of_points = [30,30,30],
	             resolution = [1.,1.,1.],
	             atomic_densities={'CA':3.5, 'CB':3.5},
	             residue_feature={
	             'PSSM' : ['./gridtest/PSSM/1CLV.protein1.ResNumPSSM','./gridtest/PSSM/1CLV.protein2.ResNumPSSM']},
	             export_path = './gridtest/input/')

#visualize the data of one complex
deeprank.map.generate_viz_files('./gridtest/')