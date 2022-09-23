from deeprankcore.models.structure import Atom, AtomicElement, Residue
from deeprankcore.models.graph import Graph


# these will ultimately be moved to deeprankcore.domain.feature.py and imported in this file.
# for now, I am keeping them here to avoid merge conflicts with the feature renaming branch.
FEATURE_NODE_ATOMICELEMENT = "atomic_element" # AtomicElement object
FEATURE_NODE_HOSTRESIDUE = "host_residue" # Residue object --> needs better name, I mean the residue that this atom belongs to
FEATURE_NODE_PDBOCCUPANCY = "pdb_occupancy" # float(0 < x < 1)



def add_features(
    graph: Graph,
):

    for node in graph.nodes:
        # not sure the block below is needed here
        atom = node.id
        residue = atom.residue

        node.features[FEATURE_NODE_ATOMICELEMENT] = atom.element
        node.features[FEATURE_NODE_HOSTRESIDUE] = residue.amino_acid
        node.features[FEATURE_NODE_PDBOCCUPANCY] = atom.occupancy
