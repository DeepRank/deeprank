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
        'pandas',
        'matplotlib',
        'torchsummary',
        'torch',
        'pdb2sql >= 0.2.1',
        'freesasa==2.0.3.post7;platform_system=="Linux"',
        'freesasa==2.0.3.post6;platform_system=="Darwin"'
        ],

    extras_require={
        'test': ['nose', 'coverage', 'pytest',
                 'pytest-cov', 'codacy-coverage', 'coveralls'],
    }
)
