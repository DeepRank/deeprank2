import os
import shutil
from tempfile import mkdtemp, mkstemp

import h5py
import numpy as np
import pytest

from deeprank2.dataset import GraphDataset, GridDataset
from deeprank2.domain import aminoacidlist as aa
from deeprank2.domain import edgestorage as Efeat
from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain import targetstorage as targets
from deeprank2.features import components, conservation, contact, surfacearea
from deeprank2.query import (
    ProteinProteinInterfaceQuery,
    QueryCollection,
    SingleResidueVariantQuery,
)
from deeprank2.utils.graph import Graph
from deeprank2.utils.grid import GridSettings, MapMethod


def _check_graph_makes_sense(
    g: Graph,
    node_feature_names: list[str],
    edge_feature_names: list[str],
) -> None:
    assert len(g.nodes) > 0, "no nodes"
    assert Nfeat.POSITION in g.nodes[0].features

    assert len(g.edges) > 0, "no edges"
    assert Efeat.DISTANCE in g.edges[0].features

    for edge in g.edges:
        if edge.id.item1 == edge.id.item2:
            msg = f"an edge pairs {edge.id.item1} with itself"
            raise ValueError(msg)

    assert not g.has_nan()

    f, tmp_path = mkstemp(suffix=".hdf5")
    os.close(f)

    try:
        g.targets[targets.BINARY] = 0
        g.write_to_hdf5(tmp_path)

        with h5py.File(tmp_path, "r") as f5:
            grp = f5[next(iter(f5.keys()))]
            for feature_name in node_feature_names:
                assert grp[f"{Nfeat.NODE}/{feature_name}"][()].size > 0, f"no {feature_name} feature"

                assert len(np.nonzero(grp[f"{Nfeat.NODE}/{feature_name}"][()])) > 0, f"{feature_name}: all zero"

            assert grp[f"{Efeat.EDGE}/{Efeat.INDEX}"][()].shape[1] == 2, "wrong edge index shape"
            assert grp[f"{Efeat.EDGE}/{Efeat.INDEX}"].shape[0] > 0, "no edge indices"

            for feature_name in edge_feature_names:
                assert (
                    grp[f"{Efeat.EDGE}/{feature_name}"][()].shape[0] == grp[f"{Efeat.EDGE}/{Efeat.INDEX}"].shape[0]
                ), f"not enough edge {feature_name} feature values"

            count_edges_hdf5 = grp[f"{Efeat.EDGE}/{Efeat.INDEX}"].shape[0]

        dataset = GraphDataset(hdf5_path=tmp_path, target=targets.BINARY)
        torch_data_entry = dataset[0]

        assert torch_data_entry is not None

        # expecting twice as many edges, because torch is directional
        count_edges_torch = torch_data_entry.edge_index.shape[1]
        assert count_edges_torch == 2 * count_edges_hdf5, f"got {count_edges_torch} edges in output data, hdf5 has {count_edges_hdf5}"

        count_edge_features_torch = torch_data_entry.edge_attr.shape[0]
        assert count_edge_features_torch == count_edges_torch, f"got {count_edge_features_torch} edge feature sets, but {count_edges_torch} edge indices"
    finally:
        os.remove(tmp_path)


def test_interface_graph_residue() -> None:
    query = ProteinProteinInterfaceQuery(
        pdb_path="tests/data/pdb/3C8P/3C8P.pdb",
        resolution="residue",
        chain_ids=["A", "B"],
        pssm_paths={
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


def test_interface_graph_atomic() -> None:
    query = ProteinProteinInterfaceQuery(
        pdb_path="tests/data/pdb/3C8P/3C8P.pdb",
        resolution="atom",
        chain_ids=["A", "B"],
        pssm_paths={
            "A": "tests/data/pssm/3C8P/3C8P.A.pdb.pssm",
            "B": "tests/data/pssm/3C8P/3C8P.B.pdb.pssm",
        },
        influence_radius=4.5,
        max_edge_length=4.5,
    )

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


def test_variant_graph_101M() -> None:
    query = SingleResidueVariantQuery(
        pdb_path="tests/data/pdb/101M/101M.pdb",
        resolution="atom",
        chain_ids="A",
        variant_residue_number=27,
        insertion_code=None,
        wildtype_amino_acid=aa.asparagine,
        variant_amino_acid=aa.phenylalanine,
        pssm_paths={"A": "tests/data/pssm/101M/101M.A.pdb.pssm"},
        targets={targets.BINARY: 0},
        influence_radius=5.0,
        max_edge_length=5.0,
    )

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


def test_variant_graph_1A0Z() -> None:
    query = SingleResidueVariantQuery(
        pdb_path="tests/data/pdb/1A0Z/1A0Z.pdb",
        resolution="atom",
        chain_ids="A",
        variant_residue_number=125,
        insertion_code=None,
        wildtype_amino_acid=aa.leucine,
        variant_amino_acid=aa.arginine,
        pssm_paths={
            "A": "tests/data/pssm/1A0Z/1A0Z.A.pdb.pssm",
            "B": "tests/data/pssm/1A0Z/1A0Z.B.pdb.pssm",
            "C": "tests/data/pssm/1A0Z/1A0Z.A.pdb.pssm",
            "D": "tests/data/pssm/1A0Z/1A0Z.B.pdb.pssm",
        },
        targets={targets.BINARY: 1},
        influence_radius=5.0,
        max_edge_length=5.0,
    )

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


def test_variant_graph_9API() -> None:
    query = SingleResidueVariantQuery(
        pdb_path="tests/data/pdb/9api/9api.pdb",
        resolution="atom",
        chain_ids="A",
        variant_residue_number=310,
        insertion_code=None,
        wildtype_amino_acid=aa.lysine,
        variant_amino_acid=aa.glutamate,
        pssm_paths={
            "A": "tests/data/pssm/9api/9api.A.pdb.pssm",
            "B": "tests/data/pssm/9api/9api.B.pdb.pssm",
        },
        targets={targets.BINARY: 0},
        influence_radius=5.0,
        max_edge_length=5.0,
    )

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


def test_variant_residue_graph_101M() -> None:
    query = SingleResidueVariantQuery(
        pdb_path="tests/data/pdb/101M/101M.pdb",
        resolution="residue",
        chain_ids="A",
        variant_residue_number=25,
        insertion_code=None,
        wildtype_amino_acid=aa.glycine,
        variant_amino_acid=aa.alanine,
        pssm_paths={"A": "tests/data/pssm/101M/101M.A.pdb.pssm"},
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


def test_res_ppi() -> None:
    query = ProteinProteinInterfaceQuery(
        pdb_path="tests/data/pdb/3MRC/3MRC.pdb",
        resolution="residue",
        chain_ids=["M", "P"],
    )
    g = query.build([surfacearea, contact])
    _check_graph_makes_sense(g, [Nfeat.SASA], [Efeat.ELEC])


def test_augmentation() -> None:
    qc = QueryCollection()

    qc.add(
        ProteinProteinInterfaceQuery(
            pdb_path="tests/data/pdb/3C8P/3C8P.pdb",
            resolution="residue",
            chain_ids=["A", "B"],
            pssm_paths={
                "A": "tests/data/pssm/3C8P/3C8P.A.pdb.pssm",
                "B": "tests/data/pssm/3C8P/3C8P.B.pdb.pssm",
            },
            targets={targets.BINARY: 0},
        ),
    )

    qc.add(
        ProteinProteinInterfaceQuery(
            pdb_path="tests/data/pdb/3C8P/3C8P.pdb",
            resolution="atom",
            chain_ids=["A", "B"],
            pssm_paths={
                "A": "tests/data/pssm/3C8P/3C8P.A.pdb.pssm",
                "B": "tests/data/pssm/3C8P/3C8P.B.pdb.pssm",
            },
            targets={targets.BINARY: 0},
        ),
    )

    qc.add(
        SingleResidueVariantQuery(
            pdb_path="tests/data/pdb/101M/101M.pdb",
            resolution="residue",
            chain_ids="A",
            variant_residue_number=25,
            insertion_code=None,
            wildtype_amino_acid=aa.glycine,
            variant_amino_acid=aa.alanine,
            pssm_paths={"A": "tests/data/pssm/101M/101M.A.pdb.pssm"},
            targets={targets.BINARY: 0},
        ),
    )

    qc.add(
        SingleResidueVariantQuery(
            pdb_path="tests/data/pdb/101M/101M.pdb",
            resolution="atom",
            chain_ids="A",
            variant_residue_number=27,
            insertion_code=None,
            wildtype_amino_acid=aa.asparagine,
            variant_amino_acid=aa.phenylalanine,
            pssm_paths={"A": "tests/data/pssm/101M/101M.A.pdb.pssm"},
            targets={targets.BINARY: 0},
            influence_radius=3.0,
        ),
    )

    augmentation_count = 3
    grid_settings = GridSettings([20, 20, 20], [20.0, 20.0, 20.0])
    expected_entry_count = (augmentation_count + 1) * len(qc)

    tmp_dir = mkdtemp()
    try:
        qc.process(
            f"{tmp_dir}/qc",
            grid_settings=grid_settings,
            grid_map_method=MapMethod.GAUSSIAN,
            grid_augmentation_count=augmentation_count,
        )

        hdf5_path = f"{tmp_dir}/qc.hdf5"
        assert os.path.isfile(hdf5_path)

        with h5py.File(hdf5_path, "r") as f5:
            entry_names = list(f5.keys())

        assert len(entry_names) == expected_entry_count, f"Found {len(entry_names)} entries, expected {expected_entry_count}"

        dataset = GridDataset(hdf5_path, target="binary")

        assert len(dataset) == expected_entry_count, f"Found {len(dataset)} data points, expected {expected_entry_count}"
    finally:
        shutil.rmtree(tmp_dir)


def test_incorrect_pssm_order() -> None:
    q = ProteinProteinInterfaceQuery(
        pdb_path="tests/data/pdb/3C8P/3C8P.pdb",
        resolution="residue",
        chain_ids=["A", "B"],
        pssm_paths={
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
        q.suppress_pssm_errors = True
        _ = q.build(conservation)


def test_incomplete_pssm() -> None:
    q = ProteinProteinInterfaceQuery(
        pdb_path="tests/data/pdb/3C8P/3C8P.pdb",
        resolution="residue",
        chain_ids=["A", "B"],
        pssm_paths={
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
        q.suppress_pssm_errors = True
        _ = q.build(conservation)


def test_no_pssm_provided() -> None:
    # pssm_paths is empty dictionary
    q_empty_dict = ProteinProteinInterfaceQuery(
        pdb_path="tests/data/pdb/3C8P/3C8P.pdb",
        resolution="residue",
        chain_ids=["A", "B"],
        pssm_paths={},
    )

    # pssm_paths not provided
    q_not_provided = ProteinProteinInterfaceQuery(
        pdb_path="tests/data/pdb/3C8P/3C8P.pdb",
        resolution="residue",
        chain_ids=["A", "B"],
    )

    with pytest.raises(ValueError):
        _ = q_empty_dict.build([conservation])
    with pytest.raises(ValueError):
        _ = q_not_provided.build([conservation])

    # no error if conservation module is not used
    _ = q_empty_dict.build([components])
    _ = q_not_provided.build([components])


def test_incorrect_pssm_provided() -> None:
    # non-existing file
    q_non_existing = ProteinProteinInterfaceQuery(
        pdb_path="tests/data/pdb/3C8P/3C8P.pdb",
        resolution="residue",
        chain_ids=["A", "B"],
        pssm_paths={
            "A": "tests/data/pssm/3C8P/3C8P.A.pdb.pssm",
            "B": "tests/data/pssm/3C8P_incorrect/dummy_non_existing_file.pssm",
        },
    )

    # missing file
    q_missing = ProteinProteinInterfaceQuery(
        pdb_path="tests/data/pdb/3C8P/3C8P.pdb",
        resolution="residue",
        chain_ids=["A", "B"],
        pssm_paths={
            "A": "tests/data/pssm/3C8P/3C8P.A.pdb.pssm",
        },
    )

    with pytest.raises(FileNotFoundError):
        _ = q_non_existing.build([conservation])
    with pytest.raises(FileNotFoundError):
        _ = q_missing.build([conservation])

    # no error if conservation module is not used
    _ = q_non_existing.build([components])
    _ = q_missing.build([components])


def test_variant_query_multiple_chains() -> None:
    q = SingleResidueVariantQuery(
        pdb_path="tests/data/pdb/2g98/pdb2g98.pdb",
        resolution="atom",
        chain_ids="A",
        variant_residue_number=14,
        insertion_code=None,
        wildtype_amino_acid=aa.arginine,
        variant_amino_acid=aa.cysteine,
        pssm_paths={"A": "tests/data/pssm/2g98/2g98.A.pdb.pssm"},
        targets={targets.BINARY: 1},
        influence_radius=10.0,
        max_edge_length=4.5,
    )

    # at radius 10, chain B is included in graph
    # no error without conservation module
    graph = q.build(components)
    assert "B" in graph.get_all_chains()
    # if we rebuild the graph with conservation module it should fail
    with pytest.raises(FileNotFoundError):
        _ = q.build(conservation)

    # at radius 7, chain B is not included in graph
    q.influence_radius = 7.0
    graph = q.build(conservation)
    assert "B" not in graph.get_all_chains()
