import os
import shutil
import tempfile
from random import randrange

import h5py
import numpy as np
import pytest
from pdb2sql import pdb2sql
from pdb2sql.transform import get_rot_axis_angle

from deeprank2.domain import edgestorage as Efeat
from deeprank2.domain import gridstorage
from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain import targetstorage as Target
from deeprank2.molstruct.pair import ResidueContact
from deeprank2.utils.buildgraph import get_structure
from deeprank2.utils.graph import Edge, Graph, Node
from deeprank2.utils.grid import Augmentation, GridSettings, MapMethod

entry_id = "test"
node_feature_narray = "node_feat1"
edge_feature_narray = "edge_feat1"
node_feature_singleton = "node_feat2"
# target name and value
target_name = "target1"
target_value = 1.0


@pytest.fixture()
def graph() -> Graph:
    """Build a simple graph of two nodes and one edge in between them."""
    # load the structure
    pdb = pdb2sql("tests/data/pdb/101M/101M.pdb")
    try:
        structure = get_structure(pdb, entry_id)
    finally:
        pdb._close()  # noqa: SLF001 (private member accessed)

    # build a contact from two residues
    residue0 = structure.chains[0].residues[0]
    residue1 = structure.chains[0].residues[1]
    contact01 = ResidueContact(residue0, residue1)

    # build two nodes and an edge
    node0 = Node(residue0)
    node1 = Node(residue1)
    edge01 = Edge(contact01)

    # add features to the nodes and edge
    node0.features[node_feature_narray] = np.array([0.1, 0.1, 0.5])
    node1.features[node_feature_narray] = np.array([1.0, 0.9, 0.5])
    edge01.features[edge_feature_narray] = np.array([2.0])
    node0.features[node_feature_singleton] = 1.0
    node1.features[node_feature_singleton] = 0.0

    # set node positions, for the grid mapping
    node0.features[Nfeat.POSITION] = residue0.get_center()
    node1.features[Nfeat.POSITION] = residue1.get_center()

    # init the graph
    graph = Graph(structure.id)
    graph.center = np.mean([node0.features[Nfeat.POSITION], node1.features[Nfeat.POSITION]], axis=0)
    graph.targets[target_name] = target_value

    graph.add_node(node0)
    graph.add_node(node1)
    graph.add_edge(edge01)
    return graph


def test_graph_write_to_hdf5(graph: Graph) -> None:
    """Test that the graph is correctly written to hdf5 file."""
    # create a temporary hdf5 file to write to
    tmp_dir_path = tempfile.mkdtemp()

    hdf5_path = os.path.join(tmp_dir_path, "101m.hdf5")

    try:
        # export graph to hdf5
        graph.write_to_hdf5(hdf5_path)

        # check the contents of the hdf5 file
        with h5py.File(hdf5_path, "r") as f5:
            grp = f5[entry_id]

            # nodes
            assert Nfeat.NODE in grp
            node_features_group = grp[Nfeat.NODE]
            assert node_feature_narray in node_features_group
            assert len(np.nonzero(node_features_group[node_feature_narray][()])) > 0
            assert node_features_group[node_feature_narray][()].shape == (2, 3)
            assert node_features_group[node_feature_singleton][()].shape == (2,)

            # edges
            assert Efeat.EDGE in grp
            edge_features_group = grp[Efeat.EDGE]
            assert edge_feature_narray in edge_features_group
            assert len(np.nonzero(edge_features_group[edge_feature_narray][()])) > 0
            assert edge_features_group[edge_feature_narray][()].shape == (1, 1)
            assert Efeat.INDEX in edge_features_group
            assert len(np.nonzero(edge_features_group[Efeat.INDEX][()])) > 0

            # target
            assert grp[Target.VALUES][target_name][()] == target_value

    finally:
        shutil.rmtree(tmp_dir_path)  # clean up after the test


def test_graph_write_as_grid_to_hdf5(graph: Graph) -> None:
    """Test that the graph is correctly written to hdf5 file as a grid."""
    # create a temporary hdf5 file to write to
    tmp_dir_path = tempfile.mkdtemp()

    hdf5_path = os.path.join(tmp_dir_path, "101m.hdf5")

    try:
        # export grid to hdf5
        grid_settings = GridSettings([20, 20, 20], [20.0, 20.0, 20.0])
        assert np.all(grid_settings.resolutions == np.array((1.0, 1.0, 1.0)))

        graph.write_as_grid_to_hdf5(hdf5_path, grid_settings, MapMethod.GAUSSIAN)

        # check the contents of the hdf5 file
        with h5py.File(hdf5_path, "r") as f5:
            grp = f5[entry_id]

            # mapped features
            assert gridstorage.MAPPED_FEATURES in grp
            mapped_group = grp[gridstorage.MAPPED_FEATURES]
            ## narray features
            for feature_name in [
                f"{node_feature_narray}_000",
                f"{node_feature_narray}_001",
                f"{node_feature_narray}_002",
                f"{edge_feature_narray}_000",
            ]:
                assert feature_name in mapped_group, f"missing mapped feature {feature_name}"
                data = mapped_group[feature_name][()]
                assert len(np.nonzero(data)) > 0, f"{feature_name}: all zero"
                assert np.all(data.shape == tuple(grid_settings.points_counts))
            ## single value features
            data = mapped_group[node_feature_singleton][()]
            assert len(np.nonzero(data)) > 0, f"{feature_name}: all zero"
            assert np.all(data.shape == tuple(grid_settings.points_counts))

            # target
            assert grp[Target.VALUES][target_name][()] == target_value

    finally:
        shutil.rmtree(tmp_dir_path)  # clean up after the test


def test_graph_augmented_write_as_grid_to_hdf5(graph: Graph) -> None:
    """Test that the graph is correctly written to hdf5 file as a grid."""
    # create a temporary hdf5 file to write to
    tmp_dir_path = tempfile.mkdtemp()

    hdf5_path = os.path.join(tmp_dir_path, "101m.hdf5")

    try:
        # export grid to hdf5
        grid_settings = GridSettings([20, 20, 20], [20.0, 20.0, 20.0])
        assert np.all(grid_settings.resolutions == np.array((1.0, 1.0, 1.0)))

        # save to hdf5
        graph.write_as_grid_to_hdf5(hdf5_path, grid_settings, MapMethod.GAUSSIAN)

        # two data points augmentation
        axis, angle = get_rot_axis_angle(randrange(100))
        augmentation = Augmentation(axis, angle)
        graph.write_as_grid_to_hdf5(hdf5_path, grid_settings, MapMethod.GAUSSIAN, augmentation)
        axis, angle = get_rot_axis_angle(randrange(100))
        augmentation = Augmentation(axis, angle)
        graph.write_as_grid_to_hdf5(hdf5_path, grid_settings, MapMethod.GAUSSIAN, augmentation)

        # check the contents of the hdf5 file
        with h5py.File(hdf5_path, "r") as f5:
            assert list(f5.keys()) == [entry_id, f"{entry_id}_000", f"{entry_id}_001"]
            grp = f5[entry_id]
            mapped_group = grp[gridstorage.MAPPED_FEATURES]
            # check that the feature value is preserved after augmentation
            unaugmented_data = mapped_group[node_feature_singleton][:]

            for aug_id in [f"{entry_id}_000", f"{entry_id}_001"]:
                grp = f5[aug_id]

                # mapped features
                assert gridstorage.MAPPED_FEATURES in grp
                mapped_group = grp[gridstorage.MAPPED_FEATURES]
                ## narray features
                for feature_name in [
                    f"{node_feature_narray}_000",
                    f"{node_feature_narray}_001",
                    f"{node_feature_narray}_002",
                    f"{edge_feature_narray}_000",
                ]:
                    assert feature_name in mapped_group, f"missing mapped feature {feature_name}"
                    data = mapped_group[feature_name][()]
                    assert len(np.nonzero(data)) > 0, f"{feature_name}: all zero"
                    assert np.all(data.shape == tuple(grid_settings.points_counts))
                ## single value features
                data = mapped_group[node_feature_singleton][()]
                assert len(np.nonzero(data)) > 0, f"{feature_name}: all zero"
                assert np.all(data.shape == tuple(grid_settings.points_counts))
                # check that the augmented data is the same, just different orientation
                assert (f5[f"{entry_id}/grid_points/center"][()] == f5[f"{aug_id}/grid_points/center"][()]).all()
                assert np.abs(np.sum(data) - np.sum(unaugmented_data)).item() < 0.2

                # target
                assert grp[Target.VALUES][target_name][()] == target_value

    finally:
        shutil.rmtree(tmp_dir_path)  # clean up after the test
