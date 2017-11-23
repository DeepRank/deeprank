#from distutils.core import setup
from setuptools import setup

setup(
    name='deeprank',
    description='Rank Protein-Protein Interactions using Deep Learning',
    version='0.1.dev0',
    url='https://github.com/DeepRank',
    packages=['deeprank'],


    install_requires=[
        'numpy >= 1.13',
        'scipy',
        'h5py',
        'tqdm',
        'matplotlib' ]
)


# numpy >= 1.13
# scipy
# h5py
# tqdm

# # tensorflow/board
# tensorflow
# tensorflow-tensorboard
# tensorboard-pytorch

# # pythroch
# http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl
# torchvision
