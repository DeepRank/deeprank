import sys
import deeprank.assemble


##########################################################################
#
#	STEP 1 ASSEMBLE THE DATA SET
#
##########################################################################

# adress of the BM4 folder
BM4 = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/'
BM4 = sys.argv[1]

# sources to assemble the data base
decoys = BM4 + 'decoys_pdbFLs/'
natives = BM4 + '/BM4_dimers_bound/pdbFLs_ori'

# the feature we want to have
features = {'ELEC'  : BM4 + '/ELEC',
			'VDW'   : BM4 + '/VDW',
			'CHARGE': BM4 + '/CHARGE'}

# the target we want to have
targets = {'haddock_score' : BM4 + '/model_qualities/haddockScore/water'}
classID = BM4 + '/training_set_IDS/classIDs.lst'

# adress of the database
database = '../../database/'

#inti the data assembler 
da = deeprank.assemble.DataAssembler(classID=classID,decoys=decoys,natives=natives,
                                     features=features,targets=targets,outdir=database,data_augmentation=10)

create new files
da.create_database()


# add a new target
targets = {'fnat' : BM4 + '/model_qualities/Fnat/water'}
da = deeprank.assemble.DataAssembler(targets=targets,outdir=database)
da.add_target()


# add a new feature
features = {'PSSM' : BM4 + '/PSSM_newformat'}
da = deeprank.assemble.DataAssembler(features=features,outdir=database)
da.add_feature()


da = deeprank.assemble.DataAssembler(outdir=database)
da.add_classID()
