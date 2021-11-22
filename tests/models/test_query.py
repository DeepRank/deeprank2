from deeprank_gnn.models.query import ProteinProteinInterfaceResidueQuery
from deeprank_gnn.models.environment import Environment
from deeprank_gnn.domain.feature import FEATURENAME_POSITION, FEATURENAME_EDGEDISTANCE


def test_interface_graph():
    environment = Environment(pdb_root="tests/data/pdb", pssm_root="tests/data/pssm", device="cpu")

    query = ProteinProteinInterfaceResidueQuery("1ATN", "A", "B")

    g = query.build_graph(environment)

    assert len(g.nodes) > 0, "no nodes"
    assert FEATURENAME_POSITION in list(g.nodes.values())[0]

    assert len(g.edges) > 0, "no edges"
    assert FEATURENAME_EDGEDISTANCE in list(g.edges.values())[0]
