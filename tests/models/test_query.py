from tempfile import mkstemp
import numpy
import os
import h5py
from deeprankcore.models.amino_acid import (
    alanine,
    arginine,
    asparagine,
    glutamate,
    glycine,
    leucine,
    lysine,
    phenylalanine
)
from deeprankcore.models.query import (
    SingleResidueVariantResidueQuery,
    SingleResidueVariantAtomicQuery,
    ProteinProteinInterfaceAtomicQuery,
    ProteinProteinInterfaceResidueQuery
)

from deeprankcore.domain.features import groups
from deeprankcore.domain.features import nodefeats as Nfeat
from deeprankcore.domain.features import edgefeats as Efeat
from deeprankcore.domain import targettypes as targets

from deeprankcore.feature import sasa, atomic_contact, bsa, pssm, amino_acid
from deeprankcore.DataSet import HDF5DataSet


def _check_graph_makes_sense(g, node_feature_names, edge_feature_names):

    assert len(g.nodes) > 0, "no nodes"
    assert groups.POSITION in g.nodes[0].features

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
                    entry_group[f"{groups.NODE}/{feature_name}"][()].size > 0
                ), f"no {feature_name} feature"

                assert (
                    len(
                        numpy.nonzero(
                            entry_group[f"{groups.NODE}/{feature_name}"][()]
                        )
                    )
                    > 0
                ), f"{feature_name}: all zero"

            assert entry_group[f"{groups.EDGE}/{groups.INDEX}"][()].shape[1] == 2, "wrong edge index shape"
            assert entry_group[f"{groups.EDGE}/{groups.INDEX}"].shape[0] > 0, "no edge indices"

            for feature_name in edge_feature_names:
                assert (
                    entry_group[f"{groups.EDGE}/{feature_name}"][()].shape[0]
                    == entry_group[f"{groups.EDGE}/{groups.INDEX}"].shape[0]
                ), f"not enough edge {feature_name} feature values"

            count_edges_hdf5 = entry_group[f"{groups.EDGE}/{groups.INDEX}"].shape[0]

        dataset = HDF5DataSet(hdf5_path=tmp_path)
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

    g = query.build_graph([bsa, amino_acid, pssm, atomic_contact])

    _check_graph_makes_sense(
        g,
        [
            groups.POSITION,
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
        interface_distance_cutoff=4.5,
    )

    # using a small cutoff here, because atomic graphs are big

    g = query.build_graph([bsa, amino_acid, pssm, atomic_contact])

    _check_graph_makes_sense(
        g,
        [
            groups.POSITION,
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
        external_distance_cutoff=5.0,
    )

    # using a small cutoff here, because atomic graphs are big

    g = query.build_graph([sasa, amino_acid, pssm, atomic_contact])

    _check_graph_makes_sense(
        g,
        [
            groups.POSITION,
            Nfeat.SASA,
            Nfeat.RESTYPE,
            Nfeat.VARIANTRES,
            Nfeat.DIFFCONSERVATION,
        ],
        [
            Efeat.DISTANCE,
            Efeat.VANDERWAALS,
            Efeat.ELECTROSTATIC,
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
        external_distance_cutoff=5.0,
        radius=5.0,
    )

    # using a small cutoff here, because atomic graphs are big

    g = query.build_graph([sasa, amino_acid, pssm, atomic_contact])

    _check_graph_makes_sense(
        g,
        [
            groups.POSITION,
            Nfeat.RESTYPE,
            Nfeat.VARIANTRES,
            Nfeat.SASA,
            Nfeat.DIFFCONSERVATION,
        ],
        [
            Efeat.DISTANCE,
            Efeat.VANDERWAALS,
            Efeat.ELECTROSTATIC,
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
        external_distance_cutoff=5.0,
        radius=5.0,
    )

    # using a small cutoff here, because atomic graphs are big

    g = query.build_graph([sasa, amino_acid, pssm, atomic_contact])

    _check_graph_makes_sense(
        g,
        [
            groups.POSITION,
            Nfeat.RESTYPE,
            Nfeat.VARIANTRES,
            Nfeat.SASA,
            Nfeat.DIFFCONSERVATION,
        ],
        [
            Efeat.DISTANCE,
            Efeat.VANDERWAALS,
            Efeat.ELECTROSTATIC,
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

    g = query.build_graph([sasa, amino_acid, pssm, atomic_contact])

    _check_graph_makes_sense(
        g,
        [
            groups.POSITION,
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

    g = query.build_graph([sasa, atomic_contact])

    _check_graph_makes_sense(g, [Nfeat.SASA], [Efeat.ELECTROSTATIC])
