# DeepRank Machinery Version 0.0

These module allows to :

   * assemble data from different sources (PDB,PSSM,Scores,....) in a comprehensible data base where each conformation has its own folder. In each folder are stored the conformation, features and targets.

   * Map several features to a grid. The type of features as well as the grid parameters can be freely chosen. New features can be mapped without havng to recompute old ones.

   * Use a 3d or 2d CNN to predict possible targets (binary class, haddock-score ...) from the data set


## Quick introdution

Minimal information to use the module 

  * clone the repository `git clone https://github.com/DeepRank/deeprank_v0.0.git`
  * go there `cd deeprank_v0.0`
  * install the module `python setup.py install`
  * download the BM4 from aclazar (/home/deep/HADDOCK-decoys/BM4_dimers)
  * go in the example folder `cd ./example/`
  * change the path to the BM4 folder in assemble.py
  * assemble the data base `python assemble.py`
  * map the features to the grid `python map.py`
  * use deep learning `python learn.py`


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

We can develop other stategies using pyMol or other in the future.

## Overview 

The (manual) workflow contains three main stages 

1 Assemble the database from a collection of sources

2 Map the features of each conformation in the database on a grid

3 Create a torch dataset from the database and use a CNN to predict a pre-defined target

The code for each stage are contained in the own folder : **_assemble/ map/ learn/_**
Examples of use for the three stages are contained in the example folder. They are detailled in the foloowing as well

## Download the data set

The docking bench mark 4 (BM4) is located on alcazar at 

```
BM4=/home/deep/HADDOCK-decoys/BM4_dimers
```

All the files needed in the following are there

  * decoys pdb : $BM4/decoys_pdbFLs

  * native pdb : $BM4/BM4_dimers_bound/pdbFLs_ori (or refined ...)

  * features   : $BM4/PSSM (only PSSM so far)

  * targets    : $BM4/model_qualities/XXX/water   (XXX=haddockscore, i-rmsd, Fnat, ....)

  * classID    : $BM4/training_set_IDS/classIDs.lst

We can later on add more features, and more targets.
The classIDs.lst contains the IDs of 228 complexes (114 natives / 114 decoys) preselected for training. The decoys were selected for their very low i-rmsd, i.e. they are very bad decoys.

Dowload the $BM4 folder (maybe zip it before as it is pretty big !)

## Assemble the database

The file assemble/assemble_data.py allows to collect data and to create a clean database. The data can contain natives pdbs, decoy pdbs, different features, different targets. In the output directory of the database, each conformation  has its own subfolder containing its pdb, features files and target data.


The example assemble.py demonstrate how to use the module to create the database. 

```python
import deeprank

# adress of the BM4 folder
BM4 = '/path/to/BM4/folder'

# sources to assemble the data base
decoys = BM4 + 'decoys_pdbFLs/'
natives = BM4 + '/BM4_dimers_bound/pdbFLs_ori'
features = {'PSSM' : BM4 + '/PSSM'}
targets = {'haddock_score' : BM4 + '/model_qualities/haddockScore/water'}
classID = BM4 + '/training_set_IDS/classIDs.lst'

# address of the database
database = './training_set/'

#init the data assembler 
da = deeprank.DataAssembler(classID=classID,decoys=decoys,natives=natives,
	              features=features,targets=targets,outdir=database)

#create new files
da.create_database()


# add a new target to the database
targets = {'fnat' : BM4 + '/model_qualities/Fnat/water'}
da = DataAssembler(targets=targets,outdir=database)
da.add_target()

# add a new feature to the database
features = {'PSSM_2' : BM4 + '/PSSM'}
da = DataAssembler(features=features,outdir=database)
da.add_feature()
```

## Map the feature to a grid

The file gridtool.py in map/ is the main class for the mapping of the features on the grid. 
This class has a lot of attributes and methods. 

The atomic densities are mapped following the protein-ligand paper. A main difference though is that the atomic density of chain A(B) are encoded as positive(negative) numbers. 


The mapping of the other features (so far only PSSM) is still very experimental. The value of the features are mapped to the grid using a bspline of degree 3. This is a usual method in Particle Mesh Ewald. 

You can test the routine on a single conformation with

```
cd $deeprankhome/map
python gridtool.py
````

This will compute the feature of the complex contained in the ./test/ subfolder. Once done go to the subfolder. Cube files can be generated using

```
python generate_cube_files.py ./test/
```

Once the cube files generated it easy to visualize them with VMD and a few files that are automatically generated by generate_cube_files.py. 

```
cd ./test/

# we must copy the pdb manually for this example
cp 1CLV_1w.pdb data_viz/complex.pdb

cd ./data_viz/

# each feature has its own vmd script file
# to visuzalize the atomiv densities use:
vmd -e AtomicDensities.vmd
```

## Map the features of all the database

The file map/gendatatool.py allows to map all the features of all the conformations contained in the database. The example file map.py shows how to use the module to do that


```python
import deeprank

# adress of the database
database = './training_set/'


#define the dictionary for the grid
#many more options are available
#see deeprank/map/gridtool.py

grid_info = {
	'atomic_densities' : {'CA':3.5,'CB':3.5,'N':3.5},
	'number_of_points' : [30,30,30],
	'resolution' : [1.,1.,1.]
}


#map the features
deeprank.map_features(database,grid_info)

#visualize the data of one complex
deeprank.generate_viz_files(database+'/1AK4')
```

After completion of the script you can visualize the atomic densities with

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

 
The example file learn.py shows how to use the module to perform deep learning

```python
import torch.optim as optim
import models3d

#adress of the database
database = './training_set/'

data_set = deeprank.DeepRankDataSet(database,
                           filter_dataset = 'decoyID.dat',
                           select_feature={'AtomicDensities' : 'all'},
                           select_target='haddock_score')

# create the network
model = deeprank.DeepRankConvNet(data_set,
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
data_set = deeprank.DeepRankDataSet(database,
                           select_feature={'AtomicDensities' : 'all'},
                           select_target='binary_class')

# create the network
model = deeprank.DeepRankConvNet(data_set,
                        models3d.ConvNet3D_binclass,
                        model_type='3d',
                        task='class',
                        tensorboard=False,
                        outdir='./test_out/')
```
After completion you should have a picture looking like that. The blue/red dots are native/deoys. The dots are in center if the CNN thinks that they are decoys and at the border if it thinks they are natives. The stars are training set, triangles validation set and circle test set. This is probably not the best way of visualizing this. Suggestions are welcome !

![alt-text](https://github.com/DeepRank/deeprank_v0.0/blob/master/pics/class_prediction.png)