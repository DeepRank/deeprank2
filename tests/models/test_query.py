import os
from tempfile import mkstemp

import h5py

from deeprank_gnn.domain.amino_acid import alanine, phenylalanine
from deeprank_gnn.models.query import ProteinProteinInterfaceResidueQuery, SingleResidueVariantAtomicQuery
from deeprank_gnn.domain.feature import FEATURENAME_POSITION, FEATURENAME_EDGEDISTANCE
from deeprank_gnn.tools.graph import graph_to_hdf5
from deeprank_gnn.DataSet import HDF5DataSet


def _check_graph_makes_sense(g, node_feature_names):

    assert len(g.nodes) > 0, "no nodes"
    assert FEATURENAME_POSITION in list(g.nodes.values())[0]

    assert len(g.edges) > 0, "no edges"
    assert FEATURENAME_EDGEDISTANCE in list(g.edges.values())[0]

    f, tmp_path = mkstemp(suffix=".hdf5")
    os.close(f)

    try:
        with h5py.File(tmp_path, 'w') as f5:
            graph_to_hdf5(g, f5)

        with h5py.File(tmp_path, 'r') as f5:
            entry_group = f5[list(f5.keys())[0]]

            for feature_name in node_feature_names:

                assert entry_group["node_data/{}".format(feature_name)][()].size > 0, "no {} feature".format(feature_name)

            assert entry_group["edge_index"][()].shape[1] == 2, "wrong edge index shape"

            assert entry_group["edge_index"].shape[0] > 0, "no edge indices"
            assert entry_group["edge_data/dist"][()].shape[0] == entry_group["edge_index"].shape[0], "not enough edge distances"

            count_edges_hdf5 = entry_group["edge_index"].shape[0]

        dataset = HDF5DataSet(database=tmp_path)
        torch_data_entry = dataset[0]

        # expecting twice as many edges, because torch is directional
        count_edges_torch = torch_data_entry.edge_index.shape[1]
        assert count_edges_torch == 2 * count_edges_hdf5, "got {} edges in output data, hdf5 has {}".format(count_edges_torch, count_edges_hdf5)

        count_edge_features_torch = torch_data_entry.edge_attr.shape[0]
        assert count_edge_features_torch == count_edges_torch, "got {} edge feature sets, but {} edge indices".format(count_edge_features_torch, count_edges_torch)
    finally:
        os.remove(tmp_path)


def test_interface_graph():
    query = ProteinProteinInterfaceResidueQuery("tests/data/pdb/1ATN/1ATN_1w.pdb", "A", "B",
                                                {"A": "tests/data/pssm/1ATN/1ATN.A.pdb.pssm",
                                                 "B": "tests/data/pssm/1ATN/1ATN.B.pdb.pssm"})

    g = query.build_graph()

    _check_graph_makes_sense(g, ["pos", "charge"])


def test_variant_graph():
    query = SingleResidueVariantAtomicQuery("tests/data/pdb/101M/101M.pdb", "A", 53, None, alanine, phenylalanine,
                                            {"A": "tests/data/pssm/101M/101M.A.pdb.pssm"},
                                            targets={"binclass": 0})

    g = query.build_graph()

    _check_graph_makes_sense(g, ["pos"])
