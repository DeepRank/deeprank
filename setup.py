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
        'matplotlib', 'torchsummary' ],

    extras_require= {
        'test': ['nose', 'coverage', 'pytest',
                 'pytest-cov','codacy-coverage','coveralls'],
    }
)
