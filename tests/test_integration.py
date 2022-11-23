from tempfile import mkdtemp
from shutil import rmtree
import warnings
import os
import h5py
from tests._utils import PATH_TEST
from deeprankcore.query import QueryCollection, ProteinProteinInterfaceResidueQuery
from deeprankcore.DataSet import HDF5DataSet
from deeprankcore.Trainer import Trainer
from deeprankcore.neuralnets.ginet import GINet
from deeprankcore.utils.metrics import OutputExporter
from deeprankcore.tools.target import compute_targets
from deeprankcore.domain import (edgestorage as Efeat, nodestorage as Nfeat,
                                targetstorage as targets)
import tempfile

def test_integration(): # pylint: disable=too-many-locals
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

    output_directory = mkdtemp()
    metrics_directory = tempfile.mkdtemp()

    prefix = os.path.join(output_directory, "test-queries-process")

    try:

        all_targets = compute_targets(pdb_path, ref_path)

        count_queries = 3
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

        output_paths = queries.process(prefix, process_count = count_queries, combine_files = False)
        assert len(output_paths) > 0

        graph_names = []
        for path in output_paths:
            with h5py.File(path, "r") as f5:
                graph_names += list(f5.keys())

        for query in queries:
            query_id = query.get_query_id()
            assert query_id in graph_names, f"missing in output: {query_id}"

        n_val = 1
        n_test = 1
        n_train = len(output_paths) - (n_val + n_test)

        node_features = [Nfeat.RESTYPE, Nfeat.POLARITY, Nfeat.BSA, Nfeat.RESDEPTH, Nfeat.HSE, Nfeat.INFOCONTENT, Nfeat.PSSM]
        edge_features = [Efeat.DISTANCE]

        dataset_train = HDF5DataSet(
            hdf5_path = output_paths[:n_train],
            node_feature = node_features,
            edge_feature = edge_features,
            target = targets.BINARY,
            clustering_method = "mcl",
        )

        dataset_val = HDF5DataSet(
            hdf5_path = output_paths[n_train:-n_test],
            node_feature = node_features,
            edge_feature = edge_features,
            target = targets.BINARY,
            clustering_method = "mcl",
        )

        dataset_test = HDF5DataSet(
            hdf5_path = output_paths[-n_test],
            node_feature = node_features,
            edge_feature = edge_features,
            target = targets.BINARY,
            clustering_method = "mcl",
        )

        trainer = Trainer(
            dataset_train,
            dataset_val,
            dataset_test,
            GINet,
            batch_size=64,
            metrics_exporters=[OutputExporter(metrics_directory)],
            transform_sigmoid=True,
        )   

        with warnings.catch_warnings(record=UserWarning):
            trainer.train(nepoch=3, validate=True) 
            trainer.save_model("test.pth.tar")

            Trainer(dataset_train, dataset_val, dataset_test, GINet, pretrained_model="test.pth.tar")

        assert len(os.listdir(metrics_directory)) > 0

    finally:
        rmtree(output_directory)
