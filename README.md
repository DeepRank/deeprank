# DeepRank

<a href='https://travis-ci.org/DeepRank/deeprank_v0.0'><img src='https://secure.travis-ci.org/DeepRank/deeprank_v0.0.png?branch=master'></a>

These module allows to :

   * Create one/multiple HDF5 files containing all the data (structures,features,mapped features,targets) required to use deep learning.

   * Train convolutional neural networks to predict possible targets (binary class, haddock-score ...) from the mapped feataures


## 1 . Installation

Minimal information to install the module 

  * clone the repository `git clone https://github.com/DeepRank/deeprank.git`
  * go there             `cd deeprank`
  * install the module   `pip install -e ./`


## 2 . Test

To test the module go to the test folder `cd ./test` and execute the following test

```
# compute atomic features (electrostatic and vdw)
python test_atomic_features.py

# compute RMSD and dockQ score
python test_rmsd.py

# generation of the data
python test_generate.py

# if you have CUDA installed 
# you can try the CUDA version
python notravis_test_generate_cuda.py

# do a bit of learning
python notravis_test_learn.py

```
 
These tests (except the notravis_* ones) are automatically run on Travis CI at each new push. So if the *build* button display *passing* they should work !

## 3 . Dependencies

### A . Requiried Dependencies

The code is written in Python3. Several packages are required to run the code but most are pretty standard. Here is an non-exhaustive list of dependencies

  * [Numpy](http://www.numpy.org)

  * [Scipy](https://www.scipy.org/)

  * [PyTorch](http://pytorch.org)

  * [h5py](http://www.h5py.org/)



The deep learning was implemented with PyTorch 2. (pytorch.org)
To install pytorch with anaconda 

```
conda install pytorch torchvision cuda80 -c soumith
```

### B . Optional dependencies

#### Network visualization

  * [tensorboard](https://github.com/lanpa/tensorboard-pytorch)

It is possible to visualize the parameters of the CNN using tensorboard. This is not required but could help in finding better architectures. You first have to install tensorflow and then install pytorch-tensorboard with pip

```
pip install tensorflow-tensorboard
pip install tensorboard-pytorch
```

### Feature visualization 

  * [VMD](http://www.ks.uiuc.edu/Research/vmd/)

The code can output visualization file of the mapped features that can be used in VMD We can develop other stategies using pyMol or other softwares in the future. All the features are exported as .cube files which is pretty standard format. The code also outputs VMD script that automatically load all the data. 

---

## 4 . Example

The `example` folder contains some script using the library. The two most important files are `generate.py` and `learn.py`.

### A . Generate the data set

The file `generate.py` contains the following:


```python
from deeprank.generate import *

# adress of the BM4 folder
BM4 = '/path/to/BM4/data/'

# sources to assemble the data base
pdb_source     = [BM4 + 'decoys_pdbFLs/1AK4/water/']
pdb_native     = [BM4 + 'BM4_dimers_bound/pdbFLs_ori']

#init the data assembler 
database = DataGenerator(pdb_source=pdb_source,
                        pdb_native=pdb_native,
                        data_augmentation=None,
                        compute_targets  = ['deeprank.tools.targets.dockQ'],
                        compute_features = ['deeprank.tools.features.atomic'],
                        hdf5='./1ak4.hdf5',
                        )

#create new files
database.create_database()

# map the features
grid_info = {
  'number_of_points' : [30,30,30],
  'resolution' : [1.,1.,1.],
  'atomic_densities' : {'CA':3.5,'N':3.5,'O':3.5,'C':3.5},
  'atomic_densities_mode' : 'diff',
  'atomic_feature'  : ['vdwaals','coulomb','charge'],
  'atomic_feature_mode': 'sum'
}

database.map_features(grid_info)
```


In  the first part of the script we define the path where to find the PDBs of the decoys and natives that we want to have in the dataset. All the .pdb files present in *pdb_source* will be used in the dataset. We need to specify where to find the native conformations to be able to compute RMSD and the dockQ score. For each pdb file detected in *pdb_source*, the code will try to find a native conformation in *pdb_native*.

We then initialize the `DataGenerator` object. This object (defined in `deeprank/generate/DataGenerator.py`) needs a few input parameters:

  * pdb_source : where to find the pdb to include in the dataset
  * pdb_native : where to find the corresponding native conformations
  * data_augmentation : None or Int. If Int=N, each molecule is duplicated N times
  * compute_targets : list of modules. Modules used to compute the targets
  * compute_features : list of modules. Modules used to compute the features
  * hdf5 : Name of the HDF5 file to store the data set

We then create the data base with the command `database.create_database()`. This function autmatically create an HDF5 files where each pdb has its own group. In each group we can find the pdb of the complex and its native form, the calculated features and the calculated targets.

We can now mapped the features to a grid. This is done via the command `database.map_features()`. As you can see this method requires a dictionary as input. The dictionary contains the instruction to map the data.

  * number_of_points: the number of points in each direction
  * resolution : the resolution in Angs
  * atomic_densities : {'atom_name' : vvdw_radius} the atomic densities required
  * atomic_densties_mode : the mapping mode of the atomic densities
  * atomic_features : the names of the atomic features we want to map
  * atomic_feature_mode : the mapping mode of the atomic features

Several features can be mapped to a grid for use as input of the deep learning phase.

**Atomic densities** The atomic densities are mapped following the [protein-ligand paper](https://arxiv.org/abs/1612.02751). 3 modes can be used to map the density of a given atom type to the grid. This can be specified through the grid **GridToolsSQL.attribute atomic_densities_mode**
  * 'diff' : density_chain_A - density_chain_B --> one grid
  * 'sum'  : density_chain_A + density_chain_B --> one grid
  * 'ind'  : density_chain_A --> one grid | density_chain_B --> one grid

**Atomic features** So far we only have the electrostatic and vdw interactions as atomic features. For each atom the value of the feature is mapped to the grid points using a Gaussian function (other modes are possible but somehow hard coded). The center of the Gaussian is the position of the atom

**Residue features** For each residue the value of the feature is mapped to the grid points using a Gaussian. The center of the Gaussian is the average position of the atoms in the residue. This is not tested throughly yet since we do not have residue features

#### Mapping with CUDA

We have recently implemented a simple CUDA kernel to use GPGPU during the mapping of the features. CUDA can therefore be enabled when mapping the features using 

```python
database.map_features(grid_info,cuda=True,gpu_block=[k,m,n])
```

By default the gpu_block is set to [8,8,8]. The Kernel Tuner can also be used to optimize the block size on the machine. This can be done with

```python
database.tune_cuda_kernel(grid_info)
```

Finally the CUDA implementation can be tested with 

```python
database.test_cuda(grid_info,gpu_block)
```

#### Visualization of the mapped features

You can find in `deeprank/tools/` a little command line utility called `visualize3dData.py` that can be used to visualize all the data mapped on the grid for a given complex. After either copyong this file in your local folder or making the original available in your path you can type

```
./visualize3Ddata.py -hdf5 1ak4.hdf5 -mol_name '1AK4_100w'
```

This will create a folder called `1AK4_100w` containing all the files needed to visualize the mapped data in VMD. Go in that folder `cd 1AK4_100w` and type

```
vmd -e AtomicDensities_diff.vmd
```

This will open VMD and load all the data and you should obtain somehting similar to the picture below:

#### Removing data from the HDF5 file

Storing all the data (structures/pdbs/grids) is great for debugging but the hdf5 file quickly becomes quite large. You can remove some data using the command lie utility `deeprank/tools/cleandata.py`. As before copy it or make the original in your path and then simply type:

```
cleandata.py 1ak4.hdf5
```

This will remove all the data non necessary for the deep learning. Hence only the mapped features and the targets values will remain in the data set. 

### 2 . Deep Learning

An example on how to use deep learning is given in `learn.py`. This file contains 


```python
from deeprank.learn import *
import model3d
import torch.optim as optim

# declare the dataset instance
database = './1ak4.hdf5'
data_set = DataSet(database,
                  select_feature={'AtomicDensities_diff' : ['C','CA','O','N'], 
                                  'atomicFeature_sum' : ['coulomb','vdwaals','charge'] },
                  select_target='DOCKQ')


# load the data set
data_set.load()

# create the network
model = ConvNet(data_set, model3d.cnn,cuda=False)

# change the optimizer (optional)
model.optimizer = optim.SGD(model.net.parameters(),
                            lr=0.001,
                            momentum=0.9,
                            weight_decay=0.005)

# start the training
model.train(nepoch = 250)
```



The first part of the script create a Torch database. The dfinition of this class is in `/deeprank/learn/DataSet.py` To do though we nee to pecify which .hdf5 file we want to use and wich features/targets included in this file we want to use during the deep learning phase. Here we specify that we want to use the the C, CA, O, and N grids included in AtomicDensities_diff and the grids coulomb,vdwaals and charge of the atomicFeature_sum. We also spcify that we want to use the DOCKQ score to train the network. More options are available to create the data set and you can read the header of the file `/deeprank/learn/DataSet.py`

We then create the ConvNet object that is defined in `/deeprank/learn/ConvNet.py`
In this minimal example we simply specify which dataset to use and wich model to use. This model is here defined in the file `model3d.py` by a class called cnn (see below). We also here turn off CUDA meaning that the training will only use CPUs. To use GPUs when available simply switch to `cuda=True`. Multiple GPUs can also be used through the options `ngpu=n`. 

We can then modify the default value for the optimizer used during the training and train the model for 250 epochs. By default the code will generate some scatter plot illustrsting the accuracy of the prediction. 


#### Model Generator

The definition of the CNN is done in `model3d.py`.

```python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils


class cnn(nn.Module):

  def __init__(self,input_shape):
    super(cnn,self).__init__()

    self.conv1 = nn.Conv3d(input_shape[0],4,kernel_size=2)
    self.pool  = nn.MaxPool3d((2,2,2))
    self.conv2 = nn.Conv3d(4,5,kernel_size=2)

    size = self._get_conv_output(input_shape)

    self.fc1   = nn.Linear(size,84)
    self.fc2   = nn.Linear(84,1)

  def _get_conv_output(self,shape):
    inp = Variable(torch.rand(1,*shape))
    out = self._forward_features(inp)
    return out.data.view(1,-1).size(1)

  def _forward_features(self,x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    return x

  def forward(self,x):
    x = self._forward_features(x)
    x = x.view(x.size(0),-1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
```


As you can see it's not trivial really. To faciliate the generation of this file we have implemented a simple model generator in `deeprank/learn/modelGenerator.py`. You can use this generator as follows:

```python
from deeprank.learn import *

conv_layers = []
conv_layers.append(conv(output_size=4,kernel_size=2,post='relu'))
conv_layers.append(pool(kernel_size=2))
conv_layers.append(conv(input_size=4,output_size=5,kernel_size=2,post='relu'))
conv_layers.append(pool(kernel_size=2))

fc_layers = []
fc_layers.append(fc(output_size=84,post='relu'))
fc_layers.append(fc(input_size=84,output_size=1))

MG = NetworkGenerator(name='cnn',fname='model.py',conv_layers=conv_layers,fc_layers=fc_layers)
MG.print()
MG.write()
```

that outputs on the screen

```
#----------------------------------------------------------------------
# Network Structure
#----------------------------------------------------------------------
#conv layer   0: conv | input -1  output  4  kernel  2  post relu
#conv layer   1: pool | kernel  2  post None
#conv layer   2: conv | input  4  output  5  kernel  2  post relu
#conv layer   3: pool | kernel  2  post None
#fc   layer   0: fc   | input -1  output  84  post relu
#fc   layer   1: fc   | input  84  output  1  post None
#----------------------------------------------------------------------
```

This defines a sinple CNN containing all the layers defined in the two list. So we first have the convolutional block that here contains a series of conv3d and pool3d and then two fully connected layers. This will automatically write a file called model.py and containing the cnn class that can be used in the training.

#### Random Model Generator

You can also get a randomly generated network with 

```python
from deeprank.learn import *
MGR = NetworkGenerator(name='cnnrand',fname='modelrandom.py')
MGR.get_new_random_model()
MGR.print()
MGR.write()
```

Here is an example of autmoatically generated network and of course the corresponding python file is also written.

```
#----------------------------------------------------------------------
# Network Structure
#----------------------------------------------------------------------
#conv layer   0: conv | input -1  output  9  kernel  4  post relu
#conv layer   1: drop | percent 0.3
#conv layer   2: drop | percent 0.4
#conv layer   3: drop | percent 0.1
#conv layer   4: pool | kernel  3  post None
#conv layer   5: conv | input  9  output  9  kernel  2  post relu
#conv layer   6: pool | kernel  2  post relu
#conv layer   7: drop | percent 0.8
#conv layer   8: conv | input  9  output  6  kernel  3  post relu
#fc   layer   0: fc   | input -1  output  1  post relu
#----------------------------------------------------------------------
```

Automatically generating models will be used for automatic hyperparameters definition in the future. 