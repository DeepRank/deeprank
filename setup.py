from distutils.core import setup

setup(
    name='deeprank',
    description='Rank Protein-Protein Interactions using Deep Learning',
    version='0.1-dev',
    url='https://github.com/DeepRank',
    package_dir = {
    'deeprank' : './',
    'deeprank.assemble' : './assemble',
    'deeprank.map'      : './map',
    'deeprank.learn'    : './learn'
    },
    packages=['deeprank',
              'deeprank.assemble','deeprank.map','deeprank.learn']
)