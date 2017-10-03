#from distutils.core import setup
from setuptools import setup

setup(
    name='deeprank',
    description='Rank Protein-Protein Interactions using Deep Learning',
    version='0.1-dev',
    url='https://github.com/DeepRank',
    package_dir = {
    'deeprank' : './',
    'deeprank.assemble' : './assemble',
    'deeprank.map'      : './map',
    'deeprank.learn'    : './learn',
    'deeprank.tools'    : './tools'
    },
    
    packages=['deeprank',
              'deeprank.assemble',
              'deeprank.map',
              'deeprank.learn',
              'deeprank.tools']
)
