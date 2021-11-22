from deeprank_gnn.models.query import ProteinProteinInterfaceResidueQuery
from deeprank_gnn.models.environment import Environment


def test_interface_graph():
    environment = Environment(pdb_root="tests/data/pdb", pssm_root="tests/data/pssm", device="cpu")

    query = ProteinProteinInterfaceResidueQuery("1ATN", "A", "B")

    g = query.build_graph(environment)

    assert len(g.nodes) > 0, "no nodes"
    assert len(g.edges) > 0, "no edges"
