#!/usr/bin/env python

import os

from setuptools import (find_packages, setup)

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, bumpver update --minor or --patch or --major
version = {}
with open(os.path.join(here, 'deeprankcore', '__init__.py')) as f:
    exec(f.read(), version)

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='deeprankcore',
    version=version['__doc__'],
    description='Graph Neural network Scoring of protein-protein conformations',
    long_description=readme + '\n\n',
    long_description_content_type='text/markdown',
    author=["Giulia Crocioni", "Coos Baakman", "Daniel Rademaker", "Gayatri Ramakrishnan", "Sven van der Burg", "Li Xue", "Daniil Lepikhov"],
    author_email='g.crocioni@esciencecenter.nl, Coos.Baakman@radboudumc.nl',
    url='https://github.com/DeepRank/deeprank-core',
    packages=find_packages(include=['deeprankcore', 'deeprankcore.*'], exclude=['tests', 'tests.*']),
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='deeprankcore',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
    test_suite='tests',

    #setup_requires=['torch>=1.11.0'],

    # not sure if the install of torch-geometric will work ..
    install_requires=[
        'numpy >= 1.21.5',
        'scipy >= 1.7.3',
        'h5py >= 3.6.0',
        'networkx', # 2.6.3
        'matplotlib', # 3.5.1
        'pdb2sql >= 0.5.1',
        'scikit-learn',
        'chart-studio', # 1.1.0
        'biopython', # 1.79
        'python-louvain', # 0.16
        'markov-clustering >= 0.0.6.dev0',
        'tqdm >= 4.63.0',
        'freesasa', # 2.1.0
        'tensorboard' # 2.9.0,
        'protobuf == 3.20.1',
        'torch-cluster>=1.6.0',
        'torch-sparse>=0.6.13',
        'torch-scatter>=2.0.9',
        'torch-spline-conv>=1.2.1'
    ],

    extras_require={
        'dev': ['yapf', 'isort'],
        'doc': ['recommonmark', 'sphinx', 'sphinx_rtd_theme'],
        'test': ['prospector[with_pyroma]', 'coverage', 'pycodestyle', 'pytest',
                 'pytest-cov', 'pytest-runner', 'coveralls'],
    })
