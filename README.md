# DeepRank Machinery Version 0.0

These files allows to :

   * assemble data from different sources (PDB,PSSM,Scores,....) in a comprehensible data base where each conformation has its own folder. In each folder are stored the conformation, features and targets.

   * Map several features to a grid. The type of features as well as the grid parameters can be freely chosen. New features can be mapped without havng to recompute old ones.

   * Use a 3d or 2d CNN to predict possible targets (binary class, haddock-score ...) from the data set

## Pre-requisite

Several packages are required to run the code. Most are pretty standard.

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

## Overview 

The (manual) workflow contains three main stages 

1 Assemble the dataset from a collection of sources
2 Map the features of each conformation of the dataset on a grid
3 Use DeepLearning to teach the CNN how to predict a pre-defined target

The code for each stage are contained in the own folder : assemble/ map/ learn/

