import tempfile
import shutil
import os
import h5py
from pdb2sql import pdb2sql
import numpy as np
from deeprankcore.utils.grid import GridSettings, MapMethod
from deeprankcore.utils.graph import Graph, Edge, Node
from deeprankcore.utils.buildgraph import get_structure
from deeprankcore.molstruct.pair import ResidueContact
from deeprankcore.domain import (edgestorage as Efeat, nodestorage as Nfeat, gridstorage)


def test_graph_build_and_export(): # pylint: disable=too-many-locals
    """Build a simple graph of two nodes and one edge in between them.
    Test that the export methods can be called without failure.
    """

    entry_id = "test"

    # load the structure
    pdb = pdb2sql("tests/data/pdb/101M/101M.pdb")
    try:
        structure = get_structure(pdb, entry_id)
    finally:
        pdb._close() # pylint: disable=protected-access

    # build a contact from two residues
    residue0 = structure.chains[0].residues[0]
    residue1 = structure.chains[0].residues[1]
    contact01 = ResidueContact(residue0, residue1)

    # build two nodes and an edge
    node0 = Node(residue0)
    node1 = Node(residue1)
    edge01 = Edge(contact01)

    # add features to the nodes and edge
    node_feature_name = "node_features"
    edge_feature_name = "edge_features"
    node_feature_singlevalue_name = "singlevalue_features"

    node0.features[node_feature_name] = np.array([0.1])
    node1.features[node_feature_name] = np.array([1.0])
    edge01.features[edge_feature_name] = np.array([2.0])
    node0.features[node_feature_singlevalue_name] = 1.0
    node1.features[node_feature_singlevalue_name] = 0.0

    # create a temporary hdf5 file to write to
    tmp_dir_path = tempfile.mkdtemp()
    hdf5_path = os.path.join(tmp_dir_path, "101m.hdf5")
    try:
        # init the graph
        graph = Graph(structure.id)

        graph.add_node(node0)
        graph.add_node(node1)
        graph.add_edge(edge01)

        # export graph to hdf5
        graph.write_to_hdf5(hdf5_path)

        # export grid to hdf5
        grid_settings = GridSettings([20, 21, 21], [20.0, 21.0, 21.0])
        assert np.all(grid_settings.resolutions == np.array((1.0, 1.0, 1.0)))

        graph.write_as_grid_to_hdf5(hdf5_path, grid_settings, MapMethod.FAST_GAUSSIAN)

        # check the contents of the hdf5 file
        with h5py.File(hdf5_path, "r") as f5:
            entry_group = f5[entry_id]

            # check for graph values
            assert Nfeat.NODE in entry_group
            node_features_group = entry_group[Nfeat.NODE]
            assert node_feature_name in node_features_group
            assert len(np.nonzero(node_features_group[node_feature_name][()])) > 0

            assert Efeat.EDGE in entry_group
            edge_features_group = entry_group[Efeat.EDGE]
            assert edge_feature_name in edge_features_group
            assert len(np.nonzero(edge_features_group[edge_feature_name][()])) > 0

            assert Efeat.INDEX in edge_features_group
            assert len(np.nonzero(edge_features_group[Efeat.INDEX][()])) > 0

            # check for grid-mapped values
            assert gridstorage.MAPPED_FEATURES in entry_group
            mapped_group = entry_group[gridstorage.MAPPED_FEATURES]

            for feature_name in (node_feature_name, edge_feature_name):
                feature_name = f"{feature_name}_000"

                assert (
                    feature_name in mapped_group
                ), f"missing mapped feature {feature_name}"
                assert "value" in mapped_group[feature_name]
                data = mapped_group[feature_name]["value"][()]
                assert len(np.nonzero(data)) > 0, f"{feature_name}: all zero"
                assert np.all(data.shape == tuple(grid_settings.points_counts))
    finally:
        shutil.rmtree(tmp_dir_path)  # clean up after the test
