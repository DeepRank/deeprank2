import numpy as np
import pytest

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain.aminoacidlist import alanine
from deeprank2.features.conservation import add_features

from . import build_testgraph


def test_conservation_residue():
    pdb_path = "tests/data/pdb/101M/101M.pdb"
    graph, variant = build_testgraph(
        pdb_path=pdb_path,
        detail="residue",
        influence_radius=10,
        max_edge_length=10,
        central_res=25,
        variant=alanine,
    )
    add_features(pdb_path, graph, variant)

    for feature_name in (
        Nfeat.PSSM,
        Nfeat.DIFFCONSERVATION,
        Nfeat.CONSERVATION,
        Nfeat.INFOCONTENT,
    ):
        assert np.any([node.features[feature_name] != 0.0 for node in graph.nodes]), f"all 0s found for {feature_name}"


def test_conservation_atom():
    pdb_path = "tests/data/pdb/101M/101M.pdb"
    graph, variant = build_testgraph(
        pdb_path=pdb_path,
        detail="atom",
        influence_radius=10,
        max_edge_length=10,
        central_res=25,
        variant=alanine,
    )
    add_features(pdb_path, graph, variant)

    for feature_name in (
        Nfeat.PSSM,
        Nfeat.DIFFCONSERVATION,
        Nfeat.CONSERVATION,
        Nfeat.INFOCONTENT,
    ):
        assert np.any([node.features[feature_name] != 0.0 for node in graph.nodes]), f"all 0s found for {feature_name}"


def test_no_pssm_file_error():
    pdb_path = "tests/data/pdb/1CRN/1CRN.pdb"
    graph, variant = build_testgraph(
        pdb_path=pdb_path,
        detail="residue",
        influence_radius=10,
        max_edge_length=10,
        central_res=17,
        variant=alanine,
    )

    with pytest.raises(FileNotFoundError):
        add_features(pdb_path, graph, variant)
