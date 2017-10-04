<<<<<<< HEAD
from distutils.core import setup
#from setuptools import setup
=======
from setuptools import setup
>>>>>>> 06c29226ddf03c767785b49224042764e9ac9e24

setup(
    name='deeprank',
    description='Rank Protein-Protein Interactions using Deep Learning',
    version='0.1-dev0',
    url='https://github.com/DeepRank',
    packages=['deeprank'],
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'matplotlib',
        'tensorboard-pytorch']
)

# setup(
#     name = "deeprank",
#     version = "0.0.1",
#     author = "Nicolas Renaud, Lars Ridder, Li Xue",
#     author_email = "n.renaud@esciencecenter.nl",
#     description = ("Rank Protein-Protein Interactions using Deep Learning"),
#     license = "BSD",
#     keywords = "deeplearning, protein docking",
#     url = "https://github.com/DeepRank",
#     packages=['assemble', 'map','learn','tools'],
#     classifiers=[
#         "Development Status :: 3 - Alpha",
#         "Topic :: Utilities",
#         "License :: OSI Approved :: BSD License",
#     ],
# )