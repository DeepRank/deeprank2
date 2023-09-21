import os
import warnings
from shutil import rmtree
from tempfile import mkdtemp

import h5py

from deeprank2.dataset import GraphDataset, GridDataset
from deeprank2.domain import edgestorage as Efeat
from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain import targetstorage as targets
from deeprank2.neuralnets.cnn.model3d import CnnClassification
from deeprank2.neuralnets.gnn.ginet import GINet
from deeprank2.query import (ProteinProteinInterfaceResidueQuery,
                             QueryCollection)
from deeprank2.tools.target import compute_ppi_scores
from deeprank2.trainer import Trainer
from deeprank2.utils.exporters import HDF5OutputExporter
from deeprank2.utils.grid import GridSettings, MapMethod

pdb_path = str("tests/data/pdb/3C8P/3C8P.pdb")
ref_path = str("tests/data/ref/3C8P/3C8P.pdb")
pssm_path1 = str("tests/data/pssm/3C8P/3C8P.A.pdb.pssm")
pssm_path2 = str("tests/data/pssm/3C8P/3C8P.B.pdb.pssm")
chain_id1 = "A"
chain_id2 = "B"

count_queries = 3


def test_cnn(): # pylint: disable=too-many-locals
    """
    Tests processing several PDB files into their features representation HDF5 file.

    Then uses HDF5 generated files to train and test a CnnRegression network.
    """

    hdf5_directory = mkdtemp()
    output_directory = mkdtemp()
    model_path = output_directory + 'test.pth.tar'

    prefix = os.path.join(hdf5_directory, "test-queries-process")

    all_targets = compute_ppi_scores(pdb_path, ref_path)

    try:
        all_targets = compute_ppi_scores(pdb_path, ref_path)

        queries = QueryCollection()
        for _ in range(count_queries):
            query = ProteinProteinInterfaceResidueQuery(
                pdb_path,
                chain_id1,
                chain_id2,
                pssm_paths={chain_id1: pssm_path1, chain_id2: pssm_path2},
                targets = all_targets
            )
            queries.add(query)

        hdf5_paths = queries.process(prefix = prefix,
                                     grid_settings=GridSettings([20, 20, 20], [20.0, 20.0, 20.0]),
                                     grid_map_method=MapMethod.GAUSSIAN)
        assert len(hdf5_paths) > 0

        graph_names = []
        for path in hdf5_paths:
            with h5py.File(path, "r") as f5:
                graph_names += list(f5.keys())

        for query in queries:
            query_id = query.get_query_id()
            assert query_id in graph_names, f"missing in output: {query_id}"

        features = [Nfeat.RESTYPE, Efeat.DISTANCE]

        dataset_train = GridDataset(
            hdf5_path = hdf5_paths,
            features = features,
            target = targets.BINARY
        )

        dataset_val = GridDataset(
            hdf5_path = hdf5_paths,
            train = False,
            dataset_train = dataset_train,
        )

        dataset_test = GridDataset(
            hdf5_path = hdf5_paths,
            train = False,
            dataset_train = dataset_train,
        )

        output_exporters = [HDF5OutputExporter(output_directory)]

        trainer = Trainer(
            CnnClassification,
            dataset_train,
            dataset_val,
            dataset_test,
            output_exporters=output_exporters
        )

        with warnings.catch_warnings(record=UserWarning):
            trainer.train(nepoch=3, batch_size=64, validate=True, best_model=False, filename=model_path)

            Trainer(CnnClassification, dataset_train, dataset_val, dataset_test, pretrained_model=model_path)

        assert len(os.listdir(output_directory)) > 0
    finally:
        rmtree(hdf5_directory)
        rmtree(output_directory)

def test_gnn(): # pylint: disable=too-many-locals
    """
    Tests processing several PDB files into their features representation HDF5 file.

    Then uses HDF5 generated files to train and test a GINet network.
    """

    hdf5_directory = mkdtemp()
    output_directory = mkdtemp()
    model_path = output_directory + 'test.pth.tar'

    prefix = os.path.join(hdf5_directory, "test-queries-process")

    try:
        all_targets = compute_ppi_scores(pdb_path, ref_path)

        queries = QueryCollection()
        for _ in range(count_queries):
            query = ProteinProteinInterfaceResidueQuery(
                pdb_path,
                chain_id1,
                chain_id2,
                pssm_paths={chain_id1: pssm_path1, chain_id2: pssm_path2},
                targets = all_targets
            )
            queries.add(query)

        hdf5_paths = queries.process(prefix = prefix)
        assert len(hdf5_paths) > 0

        graph_names = []
        for path in hdf5_paths:
            with h5py.File(path, "r") as f5:
                graph_names += list(f5.keys())

        for query in queries:
            query_id = query.get_query_id()
            assert query_id in graph_names, f"missing in output: {query_id}"

        node_features = [Nfeat.RESTYPE]
        edge_features = [Efeat.DISTANCE]


        dataset_train = GraphDataset(
            hdf5_path = hdf5_paths,
            node_features = node_features,
            edge_features = edge_features,
            clustering_method = "mcl",
            target = targets.BINARY
        )

        dataset_val = GraphDataset(
            hdf5_path = hdf5_paths,
            train = False,
            dataset_train = dataset_train,
            clustering_method = "mcl"
        )

        dataset_test = GraphDataset(
            hdf5_path = hdf5_paths,
            train = False,
            dataset_train = dataset_train,
            clustering_method = "mcl"
        )

        output_exporters = [HDF5OutputExporter(output_directory)]

        trainer = Trainer(
            GINet,
            dataset_train,
            dataset_val,
            dataset_test,
            output_exporters=output_exporters
        )

        with warnings.catch_warnings(record=UserWarning):
            trainer.train(nepoch=3, batch_size=64, validate=True, best_model=False, filename=model_path)

            Trainer(GINet, dataset_train, dataset_val, dataset_test, pretrained_model=model_path)

        assert len(os.listdir(output_directory)) > 0

    finally:
        rmtree(hdf5_directory)
        rmtree(output_directory)
