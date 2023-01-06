from tempfile import mkdtemp
from shutil import rmtree
import warnings
import os
import h5py
from tests._utils import PATH_TEST
from deeprankcore.query import QueryCollection, ProteinProteinInterfaceResidueQuery
from deeprankcore.dataset import GraphDataset, GridDataset
from deeprankcore.trainer import Trainer
from deeprankcore.neuralnets.gnn.ginet import GINet
from deeprankcore.neuralnets.cnn.model3d import CnnClassification
from deeprankcore.utils.grid import MapMethod, GridSettings
from deeprankcore.utils.exporters import HDF5OutputExporter
from deeprankcore.tools.target import compute_targets
from deeprankcore.domain import (edgestorage as Efeat, nodestorage as Nfeat,
                                targetstorage as targets)


def test_integration_cnn(): # pylint: disable=too-many-locals
    """
    Tests processing several PDB files into their features representation HDF5 file.

    Then uses HDF5 generated files to train and test a CnnRegression network.
    """

    pdb_path = str(PATH_TEST / "data/pdb/1ATN/1ATN_1w.pdb")
    ref_path = str(PATH_TEST / "data/ref/1ATN/1ATN.pdb")
    pssm_path1 = str(PATH_TEST / "data/pssm/1ATN/1ATN.A.pdb.pssm")
    pssm_path2 = str(PATH_TEST / "data/pssm/1ATN/1ATN.B.pdb.pssm")
    chain_id1 = "A"
    chain_id2 = "B"

    hdf5_directory = mkdtemp()
    output_directory = mkdtemp()

    prefix = os.path.join(hdf5_directory, "test-queries-process")

    all_targets = compute_targets(pdb_path, ref_path)

    try:
        all_targets = compute_targets(pdb_path, ref_path)

        count_queries = 5
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

        features = [Nfeat.RESTYPE, Nfeat.POLARITY, Nfeat.BSA, Nfeat.RESDEPTH, Nfeat.HSE, Nfeat.INFOCONTENT, Nfeat.PSSM, Efeat.DISTANCE]

        dataset_train = GridDataset(
            hdf5_path = hdf5_paths,
            features = features,
            target = targets.BINARY
        )

        dataset_val = GridDataset(
            hdf5_path = hdf5_paths,
            features = features,
            target = targets.BINARY
        )

        dataset_test = GridDataset(
            hdf5_path = hdf5_paths,
            features = features,
            target = targets.BINARY
        )

        output_exporters = [HDF5OutputExporter(output_directory)]

        trainer = Trainer(
            CnnClassification,
            dataset_train,
            dataset_val,
            dataset_test,
            batch_size=64,
            output_exporters=output_exporters
        )

        with warnings.catch_warnings(record=UserWarning):
            trainer.train(nepoch=3, validate=True) 
            trainer.save_model("test.pth.tar")

            Trainer(CnnClassification, dataset_train, dataset_val, dataset_test, pretrained_model="test.pth.tar")

        assert len(os.listdir(output_directory)) > 0
    finally:
        rmtree(hdf5_directory)
        rmtree(output_directory)

def test_integration_gnn(): # pylint: disable=too-many-locals
    """
    Tests processing several PDB files into their features representation HDF5 file.

    Then uses HDF5 generated files to train and test a GINet network.
    """

    pdb_path = str(PATH_TEST / "data/pdb/1ATN/1ATN_1w.pdb")
    ref_path = str(PATH_TEST / "data/ref/1ATN/1ATN.pdb")
    pssm_path1 = str(PATH_TEST / "data/pssm/1ATN/1ATN.A.pdb.pssm")
    pssm_path2 = str(PATH_TEST / "data/pssm/1ATN/1ATN.B.pdb.pssm")
    chain_id1 = "A"
    chain_id2 = "B"

    hdf5_directory = mkdtemp()
    output_directory = mkdtemp()

    prefix = os.path.join(hdf5_directory, "test-queries-process")

    try:
        all_targets = compute_targets(pdb_path, ref_path)

        count_queries = 5
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

        node_features = [Nfeat.RESTYPE, Nfeat.POLARITY, Nfeat.BSA, Nfeat.RESDEPTH, Nfeat.HSE, Nfeat.INFOCONTENT, Nfeat.PSSM]
        edge_features = [Efeat.DISTANCE]


        dataset_train = GraphDataset(
            hdf5_path = hdf5_paths,
            node_features = node_features,
            edge_features = edge_features,
            target = targets.BINARY,
            clustering_method = "mcl",
        )

        dataset_val = GraphDataset(
            hdf5_path = hdf5_paths,
            node_features = node_features,
            edge_features = edge_features,
            target = targets.BINARY,
            clustering_method = "mcl",
        )

        dataset_test = GraphDataset(
            hdf5_path = hdf5_paths,
            node_features = node_features,
            edge_features = edge_features,
            target = targets.BINARY,
            clustering_method = "mcl",
        )

        output_exporters = [HDF5OutputExporter(output_directory)]

        trainer = Trainer(
            GINet,
            dataset_train,
            dataset_val,
            dataset_test,
            batch_size=64,
            output_exporters=output_exporters
        )

        with warnings.catch_warnings(record=UserWarning):
            trainer.train(nepoch=3, validate=True) 
            trainer.save_model("test.pth.tar")

            Trainer(GINet, dataset_train, dataset_val, dataset_test, pretrained_model="test.pth.tar")

        assert len(os.listdir(output_directory)) > 0

    finally:
        rmtree(hdf5_directory)
        rmtree(output_directory)
