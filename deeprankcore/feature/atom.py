from deeprankcore.models.structure import Atom
from deeprankcore.models.graph import Graph
from deeprankcore.models.forcefield.vanderwaals import VanderwaalsParam
from deeprankcore.domain.forcefield import atomic_forcefield
from deeprankcore.models.error import UnknownAtomError
import logging


_log = logging.getLogger(__name__)


# these will ultimately be moved to deeprankcore.domain.feature.py and imported in this file.
# for now, I am keeping them here to avoid merge conflicts with the feature renaming branch.

# from models.structure.Atom
FEATURE_NODE_ATOMICELEMENT = "atomic_element" # AtomicElement object
FEATURE_NODE_PDBOCCUPANCY = "pdb_occupancy" # float(0 < x < 1)

# from feature.atomic_contact.add_features_for_atoms
FEATURE_NODE_ATOMICCHARGE = "atomic_charge" # float
FEATURE_NODE_VANDERWAALS = "vanderwaals" # ?


def add_features( # pylint: disable=unused-argument
    pdb_path: str, 
    graph: Graph, 
    *args, **kwargs
    ):

    for node in graph.nodes:
        if isinstance(node.id, Atom):   # I think this is necessary, otherwise it'll complain if it's an amino acid graph, right?
            atom = node.id

            node.features[FEATURE_NODE_ATOMICELEMENT] = atom.element.onehot
            node.features[FEATURE_NODE_PDBOCCUPANCY] = atom.occupancy
        
            try:    # I copied the try/except structure from feature.atomic_contact.add_features_for_atoms; not sure if it's actually necessary.
                node.features[FEATURE_NODE_ATOMICCHARGE] = atomic_forcefield.get_charge(atom)
                node.features[FEATURE_NODE_VANDERWAALS] = atomic_forcefield.get_vanderwaals_parameters(atom)

            except UnknownAtomError:
                _log.warning("Ignoring atom %s, because it's unknown to the forcefield", atom)

                # set parameters to zero, so that the potential becomes zero
                node.features[FEATURE_NODE_ATOMICCHARGE] = 0.0
                node.features[FEATURE_NODE_VANDERWAALS] = VanderwaalsParam(0.0, 0.0, 0.0, 0.0)
