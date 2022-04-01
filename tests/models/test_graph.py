import tempfile
import shutil
import os

from pdb2sql import pdb2sql

from deeprank_gnn.models.grid import GridSettings, MapMethod
from deeprank_gnn.models.graph import Graph, Edge, Node
from deeprank_gnn.models.contact import ResidueContact
from deeprank_gnn.tools.pdb import get_structure
from deeprank_gnn.domain.amino_acid import *



def test_graph_build_and_export():

    pdb = pdb2sql("tests/data/pdb/101M/101M.pdb")
    try:
        structure = get_structure(pdb, "101M-M0A")
    finally:
        pdb._close()

    residue0 = structure.chains[0].residues[0]
    residue1 = structure.chains[0].residues[1]
    contact01 = ResidueContact(residue0, residue1)

    node0 = Node(residue0)
    node1 = Node(residue1)
    edge01 = Edge(contact01)

    grid_settings = GridSettings(20, 20.0)
    tmp_dir_path = tempfile.mkdtemp()
    hdf5_path = os.path.join(tmp_dir_path, "101m.hdf5")
    try:
        graph = Graph(structure.id, hdf5_path)

        graph.add_node(node0)
        graph.add_node(node1)
        graph.add_edge(edge01)

        graph.to_hdf5_gnn()
        graph.to_hdf5_cnn(grid_settings, MapMethod.FAST_GAUSSIAN)
    finally:
        shutil.rmtree(tmp_dir_path)
