# DeepRank Machinery Version 0.0

These module allows to :

   * compute some features such as electrostatic interactions and van der Waals interaction 

   * assemble data from different sources (PDB,PSSM,Scores,....) in a comprehensible data base where each conformation has its own folder. In each folder are stored the conformation, features and targets.

   * Map the features to a grid. The type of features as well as the grid parameters can be freely chosen. New features can be mapped without having to recompute old ones.

   * Use a 3d or 2d CNN to predict possible targets (binary class, haddock-score ...) from the data set


## Quick introdution

Minimal information to use the module 

  * clone the repository `git clone https://github.com/DeepRank/deeprank_v0.0.git`
  * go there `cd deeprank_v0.0`
  * install the module `python setup.py install`
  * download the BM4 from aclazar (/home/deep/HADDOCK-decoys/BM4_dimers)
  * go in the example folder `cd ./example/workflow/`
  * change the path to the BM4 folder in computeFeature.py
  * compute the electrostatic and VDW interactions `python computeFeature.py`
  * change the path to the BM4 folder in assemble.py
  * assemble the data base `python assemble.py`
  * map the features to the grid `python map.py`
  * use deep learning `python learn.py`

## To Do list

There are many things that are still needed to further develop the platform. The two most important ones are:

1 *Feature Mapping* : So far we can only use

   AtomicDensities (computed on the fly)

   Atom-pair electrostatic interactions (precomputed by tools/atomicFeatures.py)

   Atom-pair van der Waals interactions (precomputed by tools/atomicFeatures.py)

   Residue PSSM (precomputed by BLAST and reformated by tools/reformat_pssm.py)


Other features can be incorporated, for example the SASA. The feature calculation must be done more generically than we do now

2 *Cuda support* : So far the code only works on CPU. We need to port them on GPUs. We can use GPUs on the gpu node of alcazar. Torch makes it relatively easy to use GPUs but the code need to be modified.

## Installation 

You should be able to install the module with

```
python setup.py install
```
If you need dependencies check the pre-requisite section. To test the installation open a python console and type

```python
import deeprank
```

## Pre-requisite

The code is written in Python3. Several packages are required to run the code but most are pretty standard. Here is an non-exhaustive list of dependencies

  * [Numpy](http://www.numpy.org)

  * [Scipy](https://www.scipy.org/)

  * [PyTorch](http://pytorch.org)



The deep learning was implemented with PyTorch 2. (pytorch.org)
To install pytorch with anaconda 

```
conda install pytorch torchvision cuda80 -c soumith
```

  * [tensorboard](https://github.com/lanpa/tensorboard-pytorch)

To install pytorch-tensorboard with pip


```
pip install tensorboard-pytorch
```

This package depends on tensorflow-tensorboard. If you don't have tensorflow installed you can get it with

```
pip install tensorflow-tensorboard
```

## Visualization of the features

The visuzalisation of the features on the grid can be done quite easily with VMD

  * [VMD](http://www.ks.uiuc.edu/Research/vmd/)

We can develop other stategies using pyMol or other softwares in the future. All the features are exported as .cube files which are pretty standard. 

---

## Feature Calculation 

In the folder example/grid/ you can find all teh files necessary to test the calculation of new features and their mapping on the grid. It uses the atmicFeature class in tools/atomic_feature.py. That class makes use of the pdb2sql method present in tools/


```python
import time
from deeprank.tools import atomicFeature

t0 = time.time()
PDB = 'complex.pdb'
FF = './forcefield/'
atfeat = atomicFeature(PDB,
                       param_charge = FF + 'protein-allhdg5-4_new.top',
                       param_vdw    = FF + 'protein-allhdg5-4_new.param',
                       patch_file   = FF + 'patch.top')

atfeat.assign_parameters()
atfeat.sqldb.prettyprint()

# compute pair interations
atfeat.evaluate_pair_interaction(print_interactions=True)

atfeat.export_data()
atfeat.sqldb.close()
print('Done in %f s' %(time.time()-t0))
```

This script outputs two files in two separated directories that contains respectively the electrostatic and vdw interactions between the contact atom of the two chains in the complex. 

## Feature Mapping and Grid Generator

 The file map/gridtool_sql.py contains the main class for generating the grid points from the PDB file and for the mapping of the features on the grid. The header of the file explains all the class attributes. One important thing is that if there is already grid points stored in the export directory, these points will be used to map the features. Hence using the class on an existing directory allows to add new features.

Several features can be mapped to a grid for use as input of the deep learning phase.

**Atomic densities** The atomic densities are mapped following the [protein-ligand paper](https://arxiv.org/abs/1612.02751). 3 modes can be used to map the density of a given atom type to the grid. This can be specified through the grid **GridToolsSQL.attribute atomic_densities_mode**
  * 'diff' : density_chain_A - density_chain_B --> one grid
  * 'sum'  : density_chain_A + density_chain_B --> one grid
  * 'ind'  : density_chain_A --> one grid
             density_chain_B --> one grid

**Atomic features** So far we only have the electrostatic and vdw interactions as atomic features. For each atom the value of the feature is mapped to the grid points using a bspline of degree 4. The center of the spline is the position of the atom

**Residue features** So far we only have the PSSM as residue features. For each residue the value of the feature is mapped to the grid points using a bspline of degree 4. The center of the spline is the average position of the atoms in the residue


The file example/grid/grid.py shows how to map the feature of one given complex to the grid. 

```python
import deeprank.map

grid = deeprank.map.GridToolsSQL(mol_name='./complex.pdb',
                     number_of_points = [30,30,30],
                     resolution = [1.,1.,1.],
                     atomic_densities={'CA':3.5, 'CB':3.5},
                     #residue_feature={
                     #'PSSM' : './PSSM/1AK4.PSSM'},
                     atomic_feature={
                     'coulomb' : './ELEC/complex.dat',
                     'vdw' : './VDW/complex.dat'
                     },
                     export_path = './input/')

#visualize the data of one complex
deeprank.map.generate_viz_files('./')

```

This will generates the grid and map the atomic densities and atomic features of the complex. The cube files and VMD script are also generated and stored in the ./data_viz subfolder. 

Once the cube files generated it easy to visualize them with VMD and a few files that are automatically generated by generate_cube_files.py. To visualize them with VMD :

```
cd ./data_viz/
vmd -e AtomicDensities.vmd
```

The data are stored in pickle files _AtomicDensity_mode.pkl_, atomicFeature.pkl and residueFeature.pkl_. Using pickle allows to easily update and/or select some data from the file using the keys of the data. 

---

## Overview of the DeepRank Worflow

The (manual) workflow contains three main stages 

0 Compute new feature if necessary

1 Assemble the database from a collection of sources

2 Map the features of each conformation in the database on a grid

3 Create a torch dataset from the database and use a CNN to predict a pre-defined target

The code for each stage are contained in the own folder : **_assemble/ map/ learn/_**
Examples of use for the three stages are contained in the example folder. They are detailled in the folowing as well. Some general tools are contained in **_ tools_**. The most important one is pdb2sql.py that allows manipulating PDB files using sqlite3 queries. 

## Download the data set

The docking bench mark 4 (BM4) is located on alcazar at 

```
BM4=/home/deep/HADDOCK-decoys/BM4_dimers
```

All the files needed in the following are there

  * decoys pdb : $BM4/decoys_pdbFLs

  * native pdb : $BM4/BM4_dimers_bound/pdbFLs_ori (or refined ...)

  * PSSM       : $BM4/PSSM_newformat

  * targets    : $BM4/model_qualities/XXX/water   (XXX=haddockscore, i-rmsd, Fnat, ....)

  * classID    : $BM4/training_set_IDS/classIDs.lst

  * Forcefield : $BM4/forcefield/

We can later on add more features, and more targets.
The classIDs.lst contains the IDs of 228 complexes (114 natives / 114 decoys) preselected for training. The decoys were selected for their very low i-rmsd, i.e. they are very bad decoys.

Dowload the $BM4 folder (maybe zip it before as it is pretty big !)

## Compute new features

New features can be calculated from the pdbs. One simple example is given in **example/worflow/computeFeature.py**. This file simply select the pdbs that are used in the DL part and compute their electrostaic and vdw features.

```python 
import os
import numpy as np
import subprocess as sp
from deeprank.tools import atomicFeature

# the root of the benchmark
BM4        = 'path/to/BM4/folder/'

# dir for writing the data
dir_elec   = BM4 + 'electrostatic/'
dir_vdw    = BM4 + 'vanDerWaals/'

# forcefield
FF         = BM4 +'./forcefield/'

# conformation
decoys     = BM4 + '/decoys_pdbFLs/'
native     = BM4 + '/BM4_dimers_bound/pdbFLs_ori'

# filter the decoys
decoyID    = './decoyID.dat'

# get the names of all the decoy pdb files in the benchmark
decoyName = sp.check_output('find %s -name  "*.pdb"' %decoys,shell=True).decode('utf-8').split()

# get the decoy ID we want to keep
decoyID = list(np.loadtxt(decoyID,str))

# filter the decy name
decoyName = [name for name in decoyName if name.split('/')[-1][:-4] in decoyID]


#get the natives names
nativeName = sp.check_output('find %s -name "*.pdb"' %native,shell=True).decode('utf-8').split()

# all the pdb we want
PDB_NAMES = nativeName + decoyName


# loop over the files
for PDB in PDB_NAMES:

        print('\nCompute Atomic Feature for %s' %(PDB.split('/')[-1][:-4]))
        atfeat = atomicFeature(
                         PDB,
             param_charge = FF + 'protein-allhdg5-4_new.top',
             param_vdw    = FF + 'protein-allhdg5-4_new.param',
             patch_file   = FF + 'patch.top',
             root_export  = BM4 )

        atfeat.assign_parameters()
        atfeat.evaluate_pair_interaction(print_interactions=False)
        atfeat.export_data()
        atfeat.sqldb.close()
```


## Assemble the database

The file assemble/assemble_data.py allows to collect data and to create a clean database. The data can contain natives pdbs, decoy pdbs, different features, different targets. In the output directory of the database, each conformation  has its own subfolder containing its pdb, features files and target data.


The example **example/worflow/assemble.py** demonstrate how to use the module to create the database. 

```python
import deeprank.assemble

BM4 = 'path/to/BM4/database/'

# sources to assemble the data base
decoys  = BM4 + '/decoys_pdbFLs/'
natives = BM4 + '/BM4_dimers_bound/pdbFLs_ori'

# the feature we want to have
features = {'ELEC' : BM4 + '/ELEC',
            'VDW'  : BM4 + '/VDW' }

# the target we want to have
targets = {'haddock_score' : BM4 + '/model_qualities/haddockScore/water'}
classID = BM4 + '/training_set_IDS/classIDs.lst'

# adress of the database
database = '../../database/'

#inti the data assembler 
da = deeprank.assemble.DataAssembler(classID=classID,decoys=decoys,natives=natives,
                                   features=features,targets=targets,outdir=database)

#create new files
da.create_database()

# add a new target
targets = {'fnat' : BM4 + '/model_qualities/Fnat/water'}
da = deeprank.assemble.DataAssembler(targets=targets,outdir=database)
da.add_target()

# add a new feature
features = {'PSSM' : BM4 + '/PSSM_newformat'}
da = deeprank.assemble.DataAssembler(features=features,outdir=database)
da.add_feature()

```

## Map the features of all the database

The file **map/gendatatool.py** allows to map all the features of all the conformations contained in the database. The example file **example/worflow/map.py** shows how to use the module to do that


```python
import deeprank.map


# adress of the database
database = '../../database/'

#define the dictionary for the grid
#many more options are available
#see deeprank/map/gridtool_sql.py

grid_info = {
  'atomic_densities' : {'CA':3.5,'CB':3.5,'N':3.5},
  'atomic_densities_mode' : 'diff',
  'number_of_points' : [30,30,30],
  #'residue_feature' : ['PSSM'],
  'atomic_feature'  : ['ELEC','VDW'],
  'resolution' : [1.,1.,1.]
}


#map the features
deeprank.map.map_features(database,grid_info)

#visualize the data of one complex
deeprank.map.generate_viz_files(database+'/1AK4')
```

In this file we map the atomic densities of CA, CB and N using a diff mode (i.e. grid = A-B). We also map the electrostatic and vdw interaction. After completion of the script you can visualize the atomic densities with

```
cd ./training_set/1AK4/data_viz
vmd -e AtomicDensities.vmd
```

You should get something that looks like that

![alt-text](https://github.com/DeepRank/deeprank_v0.0/blob/master/pics/grid.jpeg)

## Deep Learning

The main file for the deeplearning phase is learn/DeepRankConvNet.py and learn/DeepRankDataSet. A lot of options are available here. The headers f=of both file contains information about the arguments of both classes and their use. 

DeepRankDataSet allows to create a Torch data set from the database. During this process it is possible to select :

  * to select a subset of the database either with an integer or by providing a list of IDs. 
   
  * to select only some features from the database and exclude others

  * to select the target among the ones contained in the database

  * normalize the features and/or the targets

DeepRankConvnet is the class in charge of the deep learning. During the initialisation you need to provide a torch dataset outputed by DeepRankDataSet and a model, i.e. the definition of a Neural Network. Examples of such NN are provided in models2d.py and models3d.py. You can also

  * choose to use a 2d or a 3d CNN. This must match the definition of the CNN used. If a 2d CNN is chosen we can pick the plane orientation

  * choose the type of task between regression (reg) and classification (class). Depending on the task, the loss function and the plotting routines will be autmaticall adjusted.

 
The example file **example/worflow/learn.py** shows how to use the module to perform deep learning

```python
import deeprank.learn
import torch.optim as optim
import models3d

#adress of the database
database = '../../database/'

# declare the dataset instance
data_set = deeprank.learn.DeepRankDataSet(database,
                           filter_dataset = 'decoyID.dat',
                           select_feature={'AtomicDensities_diff' : ['CA','CB','N'], 
                                          'atomicFeature' : ['ELEC','VDW']},
                           select_target='haddock_score')

# Get the content of the dataset
#data_set.get_content()

# load the data set
data_set.load()

# create the network
model = deeprank.learn.DeepRankConvNet(data_set,
                        models3d.ConvNet3D_reg,
                        model_type='3d',
                        task='reg',
                        tensorboard=False,
                        outdir='./test_out/')

# change the optimizer (optional)
model.optimizer = optim.SGD(model.net.parameters(),
                            lr=0.001,
                            momentum=0.9,
                            weight_decay=0.005)

# start the training
model.train(nepoch = 250)
````

The file decoysID.dat contains only the ID of the decoys. We use this file here to filter the database. The features are all the atomic densities and the target the haddock score. 
Using this data set we perform a regression using a 3D CNN defined models3d.py. We here change the optimizer and perform only 250 epoch. After completion you should obtain a picture that looks like that

![alt-text](https://github.com/DeepRank/deeprank_v0.0/blob/master/pics/haddock_prediction.png)

To perform a regression on the binary_class using all the comformation in the database, we can modify the script as follow. Note that we use different CNN for regression and classification. The only difference is in the final layer that has 1 or  2 output.

```python
data_set = deeprank.learn.DeepRankDataSet(database,
                           select_feature={'AtomicDensities' : 'all'},
                           select_target='binary_class')

# create the network
model = deeprank.learn.DeepRankConvNet(data_set,
                        models3d.ConvNet3D_binclass,
                        model_type='3d',
                        task='class',
                        tensorboard=False,
                        outdir='./test_out/')
```
After completion you should have a picture looking like that. The blue/red dots are native/deoys. The dots are in center if the CNN thinks that they are decoys and at the border if it thinks they are natives. The stars are training set, triangles validation set and circle test set. This is probably not the best way of visualizing this. Suggestions are welcome !

![alt-text](https://github.com/DeepRank/deeprank_v0.0/blob/master/pics/class_prediction.png)