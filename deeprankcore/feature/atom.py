from deeprankcore.models.structure import Atom
from deeprankcore.models.graph import Graph
from deeprankcore.domain.forcefield import atomic_forcefield
from deeprankcore.domain.features import nodefeats
from deeprankcore.models.error import UnknownAtomError
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

            node.features[nodefeats.ATOMTYPE] = atom.element.onehot
            node.features[nodefeats.PDBOCCUPANCY] = atom.occupancy
            node.features[nodefeats.ATOMCHARGE] = atomic_forcefield.get_charge(atom)