import os
import warnings
from shutil import rmtree
from tempfile import mkdtemp

import h5py
import pandas as pd
import pytest
import torch

from deeprank2.dataset import GraphDataset, GridDataset
from deeprank2.domain import edgestorage as Efeat
from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain import targetstorage as targets
from deeprank2.neuralnets.cnn.model3d import CnnClassification
from deeprank2.neuralnets.gnn.ginet import GINet
from deeprank2.neuralnets.gnn.naive_gnn import NaiveNetwork
from deeprank2.query import ProteinProteinInterfaceQuery, QueryCollection
from deeprank2.tools.target import compute_ppi_scores
from deeprank2.trainer import Trainer
from deeprank2.utils.exporters import HDF5OutputExporter
from deeprank2.utils.grid import GridSettings, MapMethod

pdb_path = "tests/data/pdb/3C8P/3C8P.pdb"
ref_path = "tests/data/ref/3C8P/3C8P.pdb"
pssm_path1 = "tests/data/pssm/3C8P/3C8P.A.pdb.pssm"
pssm_path2 = "tests/data/pssm/3C8P/3C8P.B.pdb.pssm"
chain_id1 = "A"
chain_id2 = "B"

count_queries = 3


def test_cnn() -> None:
    """
    Tests processing several PDB files into their features representation HDF5 file.

    Then uses HDF5 generated files to train and test a CnnRegression network.
    """
    hdf5_directory = mkdtemp()
    output_directory = mkdtemp()
    model_path = output_directory + "test.pth.tar"

    prefix = os.path.join(hdf5_directory, "test-queries-process")

    try:
        all_targets = compute_ppi_scores(pdb_path, ref_path)

        queries = QueryCollection()
        for _ in range(count_queries):
            query = ProteinProteinInterfaceQuery(
                pdb_path=pdb_path,
                resolution="residue",
                chain_ids=[chain_id1, chain_id2],
                pssm_paths={chain_id1: pssm_path1, chain_id2: pssm_path2},
                targets=all_targets,
            )
            queries.add(query)

        hdf5_paths = queries.process(
            prefix=prefix,
            grid_settings=GridSettings([20, 20, 20], [20.0, 20.0, 20.0]),
            grid_map_method=MapMethod.GAUSSIAN,
        )
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
            hdf5_path=hdf5_paths,
            features=features,
            target=targets.BINARY,
        )

        dataset_val = GridDataset(
            hdf5_path=hdf5_paths,
            train_source=dataset_train,
        )

        dataset_test = GridDataset(
            hdf5_path=hdf5_paths,
            train_source=dataset_train,
        )

        output_exporters = [HDF5OutputExporter(output_directory)]

        trainer = Trainer(
            CnnClassification,
            dataset_train,
            dataset_val,
            dataset_test,
            output_exporters=output_exporters,
        )

        with warnings.catch_warnings(record=UserWarning):
            trainer.train(
                nepoch=3,
                batch_size=64,
                validate=True,
                best_model=False,
                filename=model_path,
            )

            Trainer(
                CnnClassification,
                dataset_train,
                dataset_val,
                dataset_test,
                pretrained_model=model_path,
            )

        assert len(os.listdir(output_directory)) > 0
    finally:
        rmtree(hdf5_directory)
        rmtree(output_directory)


def test_gnn() -> None:
    """Tests processing several PDB files into their features representation HDF5 file.

    Then uses HDF5 generated files to train and test a GINet network.
    """
    hdf5_directory = mkdtemp()
    output_directory = mkdtemp()
    model_path = output_directory + "test.pth.tar"

    prefix = os.path.join(hdf5_directory, "test-queries-process")

    try:
        all_targets = compute_ppi_scores(pdb_path, ref_path)

        queries = QueryCollection()
        for _ in range(count_queries):
            query = ProteinProteinInterfaceQuery(
                pdb_path=pdb_path,
                resolution="residue",
                chain_ids=[chain_id1, chain_id2],
                pssm_paths={chain_id1: pssm_path1, chain_id2: pssm_path2},
                targets=all_targets,
            )
            queries.add(query)

        hdf5_paths = queries.process(prefix=prefix)
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
            hdf5_path=hdf5_paths,
            node_features=node_features,
            edge_features=edge_features,
            clustering_method="mcl",
            target=targets.BINARY,
        )

        dataset_val = GraphDataset(
            hdf5_path=hdf5_paths,
            train_source=dataset_train,
            clustering_method="mcl",
        )

        dataset_test = GraphDataset(
            hdf5_path=hdf5_paths,
            train_source=dataset_train,
            clustering_method="mcl",
        )

        output_exporters = [HDF5OutputExporter(output_directory)]

        trainer = Trainer(
            GINet,
            dataset_train,
            dataset_val,
            dataset_test,
            output_exporters=output_exporters,
        )

        with warnings.catch_warnings(record=UserWarning):
            trainer.train(
                nepoch=3,
                batch_size=64,
                validate=True,
                best_model=False,
                filename=model_path,
            )

            Trainer(
                GINet,
                dataset_train,
                dataset_val,
                dataset_test,
                pretrained_model=model_path,
            )

        assert len(os.listdir(output_directory)) > 0

    finally:
        rmtree(hdf5_directory)
        rmtree(output_directory)


@pytest.fixture(scope="session")
def hdf5_files_for_nan(tmpdir_factory: str) -> QueryCollection:
    # For testing cases in which the loss function is nan for the validation and/or for
    # the training sets. It doesn't matter if the dataset is a GraphDataset or a GridDataset,
    # since it is a functionality of the trainer module, which does not depend on the dataset type.
    # The settings and the parameters have been carefully chosen to result in nan losses.
    pdb_paths = [
        "tests/data/pdb/3C8P/3C8P.pdb",
        "tests/data/pdb/1A0Z/1A0Z.pdb",
        "tests/data/pdb/1ATN/1ATN_1w.pdb",
    ]
    chain_id1 = "A"
    chain_id2 = "B"
    targets_values = [0, 1, 1]
    prefix = os.path.join(tmpdir_factory.mktemp("data"), "test-queries-process")

    queries = QueryCollection()
    for idx, pdb_path in enumerate(pdb_paths):
        query = ProteinProteinInterfaceQuery(
            pdb_path=pdb_path,
            resolution="residue",
            chain_ids=[chain_id1, chain_id2],
            targets={targets.BINARY: targets_values[idx]},
            # A very low radius and edge length helps for not making the network to learn
            influence_radius=3,
            max_edge_length=3,
        )
        queries.add(query)

    return queries.process(prefix=prefix)


@pytest.mark.parametrize("validate, best_model", [(True, True), (False, True), (False, False), (True, False)])  # noqa: PT006
def test_nan_loss_cases(
    validate: bool,
    best_model: bool,
    hdf5_files_for_nan,  # noqa: ANN001
) -> None:
    mols = []
    for fname in hdf5_files_for_nan:
        with h5py.File(fname, "r") as hdf5:
            for mol in hdf5:
                mols.append(mol)  # noqa: PERF402

    dataset_train = GraphDataset(
        hdf5_path=hdf5_files_for_nan,
        subset=mols[1:],
        target=targets.BINARY,
        task=targets.CLASSIF,
    )
    dataset_valid = GraphDataset(
        hdf5_path=hdf5_files_for_nan,
        subset=[mols[0]],
        train_source=dataset_train,
    )

    trainer = Trainer(NaiveNetwork, dataset_train, dataset_valid)

    optimizer = torch.optim.SGD
    lr = 10000
    weight_decay = 10000

    trainer.configure_optimizers(optimizer, lr, weight_decay=weight_decay)
    w_msg = (
        "A model has been saved but the validation and/or the training losses were NaN;\n\t"
        "try to increase the cutoff distance during the data processing or the number of data points "
        "during the training."
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        trainer.train(
            nepoch=5,
            batch_size=1,
            validate=validate,
            best_model=best_model,
            filename="test.pth.tar",
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert w_msg in str(w[-1].message)
