from deeprankcore.models.structure import Atom, AtomicElement, Residue
from deeprankcore.models.graph import Graph


# these will ultimately be moved to deeprankcore.domain.feature.py and imported in this file.
# for now, I am keeping them here to avoid merge conflicts with the feature renaming branch.

# from models.structure.Atom
FEATURE_NODE_ATOMICELEMENT = "atomic_element" # AtomicElement object
FEATURE_NODE_PDBOCCUPANCY = "pdb_occupancy" # float(0 < x < 1)

# from feature.atomic_contact.add_features_for_atoms
FEATURE_NODE_ATOMICCHARGE = "atomic_charge" # float
FEATURE_NODE_VANDERWAALS = "vanderwaals" # ?


# Do we care about residue position in an atom-graph? 
# Currently, POSITION is set to atom positions for atom graphs and average atom positions for residue-graphs
# # If so, we should create separate RESIDUEPOSITION and ATOMPOSITION features



def add_features(
    graph: Graph,
):

    for node in graph.nodes:
        if isinstance(node.id, Atom):   # I think this is necessary, otherwise it'll complain if it's an amino acid graph, right?
            atom = node.id

            node.features[FEATURE_NODE_ATOMICELEMENT] = atom.element.onehot
            node.features[FEATURE_NODE_PDBOCCUPANCY] = atom.occupancy

