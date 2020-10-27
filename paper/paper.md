---
title: 'DeepRank: A Python package for protein-protein interface calssification/ranking using 3D CNN'
tags:
  - Python
  - Deep learning
  - Protein
  - Ranking
  
authors:
  - name: Nicolas Renaud^[n.renaud@esciencecenter.nl]
    orcid: 0000-0001-9589-2694
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Cunliang Geng
    affiliation: 1
  - name: Sonja Georgievska
    affiliation: 1
  - name: Francesco Ambrosetti
    affiliation: 2
  - name: Lars Ridder
    affiliation: 1
  - name: Dario Marzella
    affiliation: 2
  - name: Alexandre Bonvin
    affiliation: 2
  - name: Li Xue
    affiliation: "2, 3"
affiliations:
 - name: Netherlands eScience Center, Science Park 140, 1098 XG, Amsterdam, The Netherlands
   index: 1
 - name: Bijvoet Centre for Biomolecular Research Facult of Science - Chemistry, Utrecht Univeristy, Padualaan 8, 3584 CH Utrecht, The Netherlands
   index: 2
 - name: Center for Molecular and Biomolecular Informatics, Radboudumc, Nijmegen, The Neterhlands
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary



# Software architecture

`DeepRank` is a Python3 pacakge build that allows for end-to-end training of neural network models on PPI data. The overall architecture of the package is shown in Fig. \autoref{fig:arch}. The package consists of two main parts: 1 - the featurization of the PPI and their mapping on 3D grid. 2 - training of 3D CNN models based on this features.

![Architecture of DeepRank.\label{fig:arch}](soft.png)

# Featurization
```python
from deeprank.generate import *
from mpi4py import MPI

# Initialize the database
database = DataGenerator( pdb_source='1AK4/decoys/', pdb_native='1AK4/native/', pssm_source='1AK4/pssm/',
    align={"selection":"interface", "plane":"xy", 'export':True}, hdf5='1ak4.hdf5', mpi_comm=MPI.COMM_WORLD)

# Compute the features and targets
database.create_database(prog_bar=True)

# Define the 3D grid
grid_info = { 'number_of_points' : [30,30,30], 'resolution' : [1.,1.,1.],
              'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}}

# Map the features
database.map_features(grid_info)
```
![Example of Featre in DeepRank.\label{fig:arch}](interface.png)

# Model Training

```python
from deeprank.learn import DataSet, NeuralNet
from deeprank.learn.model3d import cnn_reg as cnn3d

# Create data set
data_set = DataSet('1ak4.hdf5', select_target='IRMSD')

# create the network
model = NeuralNet(data_set, cnn3d, task='reg')

# start the training
model.train(nepoch=50, divide_trainset=[0.7,0.2,0.1],
            train_batch_size=5, save_model='all')
 
```
# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References