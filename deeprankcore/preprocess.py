#!/usr/bin/env python

from glob import glob
from typing import Optional, List
from functools import partial
from multiprocessing import Pool
import logging
import os

import importlib

from deeprankcore.models.query import Query


_log = logging.getLogger(__name__)


def _preprocess_one_query(prefix: str, feature_names: List[str], query: Query):

    _log.info(f'\nPreprocess query with process ID {os.getpid()}.')

    # because only one process may access an hdf5 file at the time:
    output_path = f"{prefix}-{os.getpid()}.hdf5"

    feature_modules = [importlib.import_module(name) for name in feature_names]

    graph = query.build_graph(feature_modules)

    graph.write_to_hdf5(output_path)


def preprocess(feature_modules: List, queries: List[Query],
               prefix: Optional[str] = None,
               process_count: Optional[int] = None):

    """
    Args:
        feature_modules: list of features' modules used to generate features.
        Each feature's module must implement the add_features function, and
        features' modules can be found (or should be placed in case of a custom made feature)
        in deeprankcore.feature folder.

        queries: all the queries objects that have to be preprocessed.

        prefix: prefix for the output files. ./preprocessed-data- by default.
        
        process_count: how many subprocesses to be run simultaneously.
        By default takes all available cpu cores.
    """

    if process_count is None:
        # returns the set of CPUs available considering the sched_setaffinity Linux system call,
        # which limits which CPUs a process and its children can run on.
        process_count = len(os.sched_getaffinity(0))

    _log.info(f'\nSet of CPU processors available: {process_count}.')

    if prefix is None:
        prefix = "preprocessed-data"

    _log.info('Creating pool function to process the queries...')
    pool_function = partial(_preprocess_one_query, prefix,
                            [m.__name__ for m in feature_modules])

    with Pool(process_count) as pool:
        _log.info('Starting pooling...\n')
        pool.map(pool_function, queries)

    output_paths = glob(f"{prefix}-*.hdf5")
    return output_paths
