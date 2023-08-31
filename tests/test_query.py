import os
import shutil
from tempfile import mkdtemp, mkstemp

import h5py
import numpy as np
import pytest
from deeprank2.dataset import GraphDataset, GridDataset
from deeprank2.domain.aminoacidlist import (alanine, arginine, asparagine,
                                            cysteine, glutamate, glycine,
                                            leucine, lysine, phenylalanine)
from deeprank2.query import (ProteinProteinInterfaceAtomicQuery,
                             ProteinProteinInterfaceResidueQuery,
                             QueryCollection, SingleResidueVariantAtomicQuery,
                             SingleResidueVariantResidueQuery)
from deeprank2.utils.grid import GridSettings, MapMethod

from deeprank2.domain import edgestorage as Efeat
from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain import targetstorage as targets
from deeprank2.features import components, conservation, contact, surfacearea


def _check_graph_makes_sense(g, node_feature_names, edge_feature_names):

    assert len(g.nodes) > 0, "no nodes"
    assert Nfeat.POSITION in g.nodes[0].features

    assert len(g.edges) > 0, "no edges"
    assert Efeat.DISTANCE in g.edges[0].features

    for edge in g.edges:
        if edge.id.item1 == edge.id.item2:
            raise ValueError(f"an edge pairs {edge.id.item1} with itself")

    assert not g.has_nan()

    f, tmp_path = mkstemp(suffix=".hdf5")
    os.close(f)

    try:
        g.write_to_hdf5(tmp_path)

        with h5py.File(tmp_path, "r") as f5:
            entry_group = f5[list(f5.keys())[0]]
            for feature_name in node_feature_names:
                assert (
                    entry_group[f"{Nfeat.NODE}/{feature_name}"][()].size > 0
                ), f"no {feature_name} feature"

                assert (
                    len(
                        np.nonzero(
                            entry_group[f"{Nfeat.NODE}/{feature_name}"][()]
                        )
                    )
                    > 0
                ), f"{feature_name}: all zero"

            assert entry_group[f"{Efeat.EDGE}/{Efeat.INDEX}"][()].shape[1] == 2, "wrong edge index shape"
            assert entry_group[f"{Efeat.EDGE}/{Efeat.INDEX}"].shape[0] > 0, "no edge indices"

            for feature_name in edge_feature_names:
                assert (
                    entry_group[f"{Efeat.EDGE}/{feature_name}"][()].shape[0]
                    == entry_group[f"{Efeat.EDGE}/{Efeat.INDEX}"].shape[0]
                ), f"not enough edge {feature_name} feature values"

            count_edges_hdf5 = entry_group[f"{Efeat.EDGE}/{Efeat.INDEX}"].shape[0]

        dataset = GraphDataset(hdf5_path=tmp_path)
        torch_data_entry = dataset[0]
        assert torch_data_entry is not None

        # expecting twice as many edges, because torch is directional
        count_edges_torch = torch_data_entry.edge_index.shape[1]
        assert (
            count_edges_torch == 2 * count_edges_hdf5
        ), f"got {count_edges_torch} edges in output data, hdf5 has {count_edges_hdf5}"

        count_edge_features_torch = torch_data_entry.edge_attr.shape[0]
        assert (
            count_edge_features_torch == count_edges_torch
        ), f"got {count_edge_features_torch} edge feature sets, but {count_edges_torch} edge indices"
    finally:
        os.remove(tmp_path)


def test_interface_graph_residue():
    query = ProteinProteinInterfaceResidueQuery(
        "tests/data/pdb/3C8P/3C8P.pdb",
        "A",
        "B",
        {
            "A": "tests/data/pssm/3C8P/3C8P.A.pdb.pssm",
            "B": "tests/data/pssm/3C8P/3C8P.B.pdb.pssm",
        },
    )

    g = query.build([surfacearea, components, conservation, contact])

    _check_graph_makes_sense(
        g,
        [
            Nfeat.POSITION,
            Nfeat.POLARITY,
            Nfeat.PSSM,
            Nfeat.INFOCONTENT,
        ],
        [Efeat.DISTANCE],
    )


def test_interface_graph_atomic():
    query = ProteinProteinInterfaceAtomicQuery(
        "tests/data/pdb/3C8P/3C8P.pdb",
        "A",
        "B",
        {
            "A": "tests/data/pssm/3C8P/3C8P.A.pdb.pssm",
            "B": "tests/data/pssm/3C8P/3C8P.B.pdb.pssm",
        },
        distance_cutoff=4.5,
    )

    # using a small cutoff here, because atomic graphs are big

    g = query.build([surfacearea, components, conservation, contact])

    _check_graph_makes_sense(
        g,
        [
            Nfeat.POSITION,
            Nfeat.PSSM,
            Nfeat.BSA,
            Nfeat.INFOCONTENT,
        ],
        [Efeat.DISTANCE],
    )


def test_variant_graph_101M():
    query = SingleResidueVariantAtomicQuery(
        "tests/data/pdb/101M/101M.pdb",
        "A",
        27,
        None,
        asparagine,
        phenylalanine,
        {"A": "tests/data/pssm/101M/101M.A.pdb.pssm"},
        targets={targets.BINARY: 0},
        radius=5.0,
        distance_cutoff=5.0,
    )

    # using a small cutoff here, because atomic graphs are big

    g = query.build([surfacearea, components, conservation, contact])

    _check_graph_makes_sense(
        g,
        [
            Nfeat.POSITION,
            Nfeat.SASA,
            Nfeat.RESTYPE,
            Nfeat.VARIANTRES,
            Nfeat.DIFFCONSERVATION,
        ],
        [
            Efeat.DISTANCE,
            Efeat.VDW,
            Efeat.ELEC,
        ],
    )


def test_variant_graph_1A0Z():
    query = SingleResidueVariantAtomicQuery(
        "tests/data/pdb/1A0Z/1A0Z.pdb",
        "A",
        125,
        None,
        leucine,
        arginine,
        {
            "A": "tests/data/pssm/1A0Z/1A0Z.A.pdb.pssm",
            "B": "tests/data/pssm/1A0Z/1A0Z.B.pdb.pssm",
            "C": "tests/data/pssm/1A0Z/1A0Z.A.pdb.pssm",
            "D": "tests/data/pssm/1A0Z/1A0Z.B.pdb.pssm",
        },
        targets={targets.BINARY: 1},
        distance_cutoff=5.0,
        radius=5.0,
    )

    # using a small cutoff here, because atomic graphs are big

    g = query.build([surfacearea, components, conservation, contact])

    _check_graph_makes_sense(
        g,
        [
            Nfeat.POSITION,
            Nfeat.RESTYPE,
            Nfeat.VARIANTRES,
            Nfeat.SASA,
            Nfeat.DIFFCONSERVATION,
        ],
        [
            Efeat.DISTANCE,
            Efeat.VDW,
            Efeat.ELEC,
        ],
    )


def test_variant_graph_9API():
    query = SingleResidueVariantAtomicQuery(
        "tests/data/pdb/9api/9api.pdb",
        "A",
        310,
        None,
        lysine,
        glutamate,
        {
            "A": "tests/data/pssm/9api/9api.A.pdb.pssm",
            "B": "tests/data/pssm/9api/9api.B.pdb.pssm",
        },
        targets={targets.BINARY: 0},
        distance_cutoff=5.0,
        radius=5.0,
    )

    # using a small cutoff here, because atomic graphs are big

    g = query.build([surfacearea, components, conservation, contact])

    _check_graph_makes_sense(
        g,
        [
            Nfeat.POSITION,
            Nfeat.RESTYPE,
            Nfeat.VARIANTRES,
            Nfeat.SASA,
            Nfeat.DIFFCONSERVATION,
        ],
        [
            Efeat.DISTANCE,
            Efeat.VDW,
            Efeat.ELEC,
        ],
    )


def test_variant_residue_graph_101M():
    query = SingleResidueVariantResidueQuery(
        "tests/data/pdb/101M/101M.pdb",
        "A",
        25,
        None,
        glycine,
        alanine,
        {"A": "tests/data/pssm/101M/101M.A.pdb.pssm"},
        targets={targets.BINARY: 0},
    )

    g = query.build([surfacearea, components, conservation, contact])

    _check_graph_makes_sense(
        g,
        [
            Nfeat.POSITION,
            Nfeat.SASA,
            Nfeat.PSSM,
            Nfeat.RESTYPE,
            Nfeat.VARIANTRES,
            Nfeat.POLARITY,
        ],
        [Efeat.DISTANCE],
    )


def test_res_ppi():

    query = ProteinProteinInterfaceResidueQuery("tests/data/pdb/3MRC/3MRC.pdb",
                                                "M", "P")

    g = query.build([surfacearea, contact])

    _check_graph_makes_sense(g, [Nfeat.SASA], [Efeat.ELEC])


def test_augmentation():
    qc = QueryCollection()

    qc.add(ProteinProteinInterfaceResidueQuery(
        "tests/data/pdb/3C8P/3C8P.pdb",
        "A",
        "B",
        {
            "A": "tests/data/pssm/3C8P/3C8P.A.pdb.pssm",
            "B": "tests/data/pssm/3C8P/3C8P.B.pdb.pssm",
        },
        targets={targets.BINARY: 0},
    ))

    qc.add(ProteinProteinInterfaceAtomicQuery(
        "tests/data/pdb/3C8P/3C8P.pdb",
        "A",
        "B",
        {
            "A": "tests/data/pssm/3C8P/3C8P.A.pdb.pssm",
            "B": "tests/data/pssm/3C8P/3C8P.B.pdb.pssm",
        },
        targets={targets.BINARY: 0},
    ))

    qc.add(SingleResidueVariantResidueQuery(
        "tests/data/pdb/101M/101M.pdb",
        "A",
        25,
        None,
        glycine,
        alanine,
        {"A": "tests/data/pssm/101M/101M.A.pdb.pssm"},
        targets={targets.BINARY: 0},
    ))

    qc.add(SingleResidueVariantAtomicQuery(
        "tests/data/pdb/101M/101M.pdb",
        "A",
        27,
        None,
        asparagine,
        phenylalanine,
        {"A": "tests/data/pssm/101M/101M.A.pdb.pssm"},
        targets={targets.BINARY: 0},
        radius=3.0,
    ))

    augmentation_count = 3
    grid_settings = GridSettings([20, 20, 20], [20.0, 20.0, 20.0])

    expected_entry_count = (augmentation_count + 1) * len(qc)

    tmp_dir = mkdtemp()
    try:
        qc.process(f"{tmp_dir}/qc",
                   grid_settings=grid_settings,
                   grid_map_method=MapMethod.GAUSSIAN,
                   grid_augmentation_count=augmentation_count)

        hdf5_path = f"{tmp_dir}/qc.hdf5"
        assert os.path.isfile(hdf5_path)

        with h5py.File(hdf5_path, 'r') as f5:
            entry_names = list(f5.keys())

        assert len(entry_names) == expected_entry_count, f"Found {len(entry_names)} entries, expected {expected_entry_count}"

        dataset = GridDataset(hdf5_path)

        assert len(dataset) == expected_entry_count, f"Found {len(dataset)} data points, expected {expected_entry_count}"
    finally:
        shutil.rmtree(tmp_dir)


def test_incorrect_pssm_order():
    q = ProteinProteinInterfaceResidueQuery(
        "tests/data/pdb/3C8P/3C8P.pdb",
        "A",
        "B",
        {
            "A": "tests/data/pssm/3C8P_incorrect/3C8P.A.wrong_order.pdb.pssm",
            "B": "tests/data/pssm/3C8P/3C8P.B.pdb.pssm",
        },
    )

    # check that error is thrown for incorrect pssm
    with pytest.raises(ValueError):
        _ = q.build(conservation)

    # no error if conservation module is not used
    _ = q.build(components)

    # check that error suppression works
    with pytest.warns(UserWarning):
        q._suppress = True  # pylint: disable = protected-access
        _ = q.build(conservation)


def test_incomplete_pssm():
    q = ProteinProteinInterfaceResidueQuery(
        "tests/data/pdb/3C8P/3C8P.pdb",
        "A",
        "B",
        {
            "A": "tests/data/pssm/3C8P/3C8P.A.pdb.pssm",
            "B": "tests/data/pssm/3C8P_incorrect/3C8P.B.missing_res.pdb.pssm",
        },
    )

    with pytest.raises(ValueError):
        _ = q.build(conservation)

    # no error if conservation module is not used
    _ = q.build(components)

    # check that error suppression works
    with pytest.warns(UserWarning):
        q._suppress = True  # pylint: disable = protected-access
        _ = q.build(conservation)


def test_no_pssm_provided():
    # pssm_paths is empty dictionary
    q_empty_dict = ProteinProteinInterfaceResidueQuery(
        "tests/data/pdb/3C8P/3C8P.pdb",
        "A",
        "B",
        {},
    )

    # pssm_paths not provided
    q_not_provided = ProteinProteinInterfaceResidueQuery(
        "tests/data/pdb/3C8P/3C8P.pdb",
        "A",
        "B",
    )

    with pytest.raises(ValueError):
        _ = q_empty_dict.build(conservation)
        _ = q_not_provided.build(conservation)

    # no error if conservation module is not used
    _ = q_empty_dict.build(components)
    _ = q_not_provided.build(components)


def test_incorrect_pssm_provided():
    # non-existing file
    q_non_existing = ProteinProteinInterfaceResidueQuery(
        "tests/data/pdb/3C8P/3C8P.pdb",
        "A",
        "B",
        {
            "A": "tests/data/pssm/3C8P/3C8P.A.pdb.pssm",
            "B": "tests/data/pssm/3C8P_incorrect/dummy_non_existing_file.pssm",
        },
    )

    # missing file
    q_missing = ProteinProteinInterfaceResidueQuery(
        "tests/data/pdb/3C8P/3C8P.pdb",
        "A",
        "B",
        {
            "A": "tests/data/pssm/3C8P/3C8P.A.pdb.pssm",
        },
    )

    with pytest.raises(FileNotFoundError):
        _ = q_non_existing.build(conservation)
        _ = q_missing.build(conservation)

    # no error if conservation module is not used
    _ = q_non_existing.build(components)
    _ = q_missing.build(components)


def test_variant_query_multiple_chains():
    q = SingleResidueVariantAtomicQuery(
        pdb_path = "tests/data/pdb/2g98/pdb2g98.pdb",
        chain_id = "A",
        residue_number = 14,
        insertion_code = None,
        wildtype_amino_acid = arginine,
        variant_amino_acid = cysteine,
        pssm_paths = {"A": "tests/data/pssm/2g98/2g98.A.pdb.pssm"},
        targets = {targets.BINARY: 1},
        radius = 10.0,
        distance_cutoff = 4.5,
        )

    # at radius 10, chain B is included in graph
    # no error without conservation module
    graph = q.build(components)
    assert 'B' in graph.get_all_chains()
    # if we rebuild the graph with conservation module it should fail
    with pytest.raises(FileNotFoundError):
        _ = q.build(conservation)

    # at radius 7, chain B is not included in graph
    q._radius = 7.0  # pylint: disable = protected-access
    graph = q.build(conservation)
    assert 'B' not in graph.get_all_chains()
