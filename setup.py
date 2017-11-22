#from distutils.core import setup
from setuptools import setup

setup(
    name='deeprank',
    description='Rank Protein-Protein Interactions using Deep Learning',
    version='0.1.dev0',
    url='https://github.com/DeepRank',
    packages=['deeprank'],
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'matplotlib',
        'tensorboard-pytorch']
)
