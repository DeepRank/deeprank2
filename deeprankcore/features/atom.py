from deeprankcore.models.structure import Atom
from deeprankcore.models.graph import Graph
from deeprankcore.domain.forcefield import atomic_forcefield
from deeprankcore.domain import nodefeatures
import logging


_log = logging.getLogger(__name__)


def add_features( # pylint: disable=unused-argument
    pdb_path: str, 
    graph: Graph, 
    *args, **kwargs
    ):

    for node in graph.nodes:
        if isinstance(node.id, Atom):
            atom = node.id

            node.features[nodefeatures.ATOMTYPE] = atom.element.onehot
            node.features[nodefeatures.PDBOCCUPANCY] = atom.occupancy
            node.features[nodefeatures.ATOMCHARGE] = atomic_forcefield.get_charge(atom)