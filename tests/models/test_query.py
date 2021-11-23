import os
from tempfile import mkstemp

import h5py

from deeprank_gnn.models.query import ProteinProteinInterfaceResidueQuery
from deeprank_gnn.models.environment import Environment
from deeprank_gnn.domain.feature import FEATURENAME_POSITION, FEATURENAME_EDGEDISTANCE
from deeprank_gnn.tools.graph import graph_to_hdf5


def test_interface_graph():
    environment = Environment(pdb_root="tests/data/pdb", pssm_root="tests/data/pssm", device="cpu")

    query = ProteinProteinInterfaceResidueQuery("1ATN", "A", "B")

    g = query.build_graph(environment)

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

            assert entry_group["node_data/pos"][()].size > 0, "no position feature"
            assert entry_group["node_data/charge"][()].size > 0, "no charge feature"

            assert entry_group["edge_index"][()].shape[1] == 2, "wrong edge index shape"

            assert entry_group["edge_index"].shape[0] > 0, "no edge indices"
            assert entry_group["edge_data/dist"][()].shape[0] == entry_group["edge_index"].shape[0], "not enough edge distances"

    finally:
        os.remove(tmp_path)
