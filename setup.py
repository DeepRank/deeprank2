#!/usr/bin/env python

import os

from setuptools import (find_packages, setup)

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit DeepRank-GNN/__version__.py
version = {}
with open(os.path.join(here, 'deeprank_gnn', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='DeepRank-GNN-2',
    version=version['__version__'],
    description='Graph Neural network Scoring of protein-protein conformations',
    long_description=readme + '\n\n',
    long_description_content_type='text/markdown',
    author=["Giulia Crocioni", "Coos Baakman", "Daniel Rademaker", "Gayatri Ramakrishnan", "Sven van der Burg", "Li Xue", "Daniil Lepikhov"],
    author_email='g.crocioni@esciencecenter.nl, Coos.Baakman@radboudumc.nl',
    url='https://github.com/DeepRank/deeprank-gnn-2',
    packages=find_packages(),
    package_dir={'deeprank_gnn': 'deeprank_gnn'},
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='deeprank_gnn_2',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English', 'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
    test_suite='tests',

    # not sure if the install of torch-geometric will work ..
    install_requires=[
        'numpy >= 1.13', 'scipy', 'h5py', 'torch>=1.5.0', 'networkx', 'matplotlib',
        'pdb2sql', 'sklearn', 'chart-studio', 'BioPython', 'python-louvain',
        'markov-clustering', 'torch-sparse', 'torch-scatter', 'torch-cluster',
        'torch-spline-conv', 'torch-geometric', 'tqdm', 'freesasa'
    ],
    extras_require={
        'dev': ['prospector[with_pyroma]', 'yapf', 'isort'],
        'doc': ['recommonmark', 'sphinx', 'sphinx_rtd_theme'],
        'test':
        ['coverage', 'pycodestyle', 'pytest',
            'pytest-cov', 'pytest-runner', 'coveralls'],
    })
