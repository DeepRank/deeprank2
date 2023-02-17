from typing import List
import logging
import warnings
import numpy as np
from scipy.spatial import distance_matrix
from deeprankcore.molstruct.atom import Atom
from deeprankcore.utils.graph import Graph
from deeprankcore.molstruct.pair import ResidueContact, AtomicContact
from deeprankcore.domain import edgestorage as Efeat
from deeprankcore.utils.parsing import atomic_forcefield

_log = logging.getLogger(__name__)

MAX_COVALENT_DISTANCE = 2.1

def _get_coulomb_potentials(atoms: List[Atom], distances: np.ndarray) -> np.ndarray:
    """ Calculate Coulomb potentials between between all Atoms in atom.

    Warning: there's no distance cutoff here. The radius of influence is assumed to infinite.
    However, the potential tends to 0 at large distance.
    """

    EPSILON0 = 1.0
    COULOMB_CONSTANT = 332.0636
    charges = [atomic_forcefield.get_charge(atom) for atom in atoms]
    coulomb_potentials = np.expand_dims(charges, axis=1) * np.expand_dims(charges, axis=0) * COULOMB_CONSTANT / (EPSILON0 * distances)
    return coulomb_potentials


def _get_lennard_jones_potentials(atoms: List[Atom], distances: np.ndarray) -> np.ndarray:
    """ Calculate Lennard-Jones potentials between all Atoms in atom.

    Warning: there's no distance cutoff here. The radius of influence is assumed to infinite.
    However, the potential tends to 0 at large distance.
    """

    # calculate intra potentials
    sigmas = [atomic_forcefield.get_vanderwaals_parameters(atom).intra_sigma for atom in atoms]
    epsilons = [atomic_forcefield.get_vanderwaals_parameters(atom).intra_epsilon for atom in atoms]
    mean_sigmas = 0.5 * np.add.outer(sigmas,sigmas)
    geomean_eps = np.sqrt(np.multiply.outer(epsilons,epsilons))     # sqrt(eps1*eps2)
    intra_potentials = 4.0 * geomean_eps * ((mean_sigmas / distances) ** 12 - (mean_sigmas / distances) ** 6)
    
    # calculate inter potentials
    sigmas = [atomic_forcefield.get_vanderwaals_parameters(atom).inter_sigma for atom in atoms]
    epsilons = [atomic_forcefield.get_vanderwaals_parameters(atom).inter_epsilon for atom in atoms]
    mean_sigmas = 0.5 * np.add.outer(sigmas,sigmas)
    geomean_eps = np.sqrt(np.multiply.outer(epsilons,epsilons))     # sqrt(eps1*eps2)
    inter_potentials = 4.0 * geomean_eps * ((mean_sigmas / distances) ** 12 - (mean_sigmas / distances) ** 6)

    lennard_jones_potentials = {'intra': intra_potentials, 'inter': inter_potentials}
    return lennard_jones_potentials


def add_features(pdb_path: str, graph: Graph, *args, **kwargs): # pylint: disable=too-many-locals, unused-argument
    # assign each atoms (from all edges) a unique index
    all_atoms = set() 
    if isinstance(graph.edges[0].id, AtomicContact):
        for edge in graph.edges:
            contact = edge.id
            all_atoms.add(contact.atom1)
            all_atoms.add(contact.atom2)
    elif isinstance(graph.edges[0].id, ResidueContact):
        for edge in graph.edges:
            contact = edge.id
            for atom in (contact.residue1.atoms + contact.residue2.atoms):
                all_atoms.add(atom)
    else:
        raise TypeError(
            f"Unexpected edge type: {type(graph.edges[0].id)}")
    all_atoms = list(all_atoms)
    atom_dict = {all_atoms[i]: i for i in range(len(all_atoms))}

    # make pairwise calculations between all atoms in the set
    with warnings.catch_warnings(record=RuntimeWarning):
        warnings.simplefilter("ignore")
        positions = [atom.position for atom in all_atoms]
        interatomic_distances = distance_matrix(positions, positions)
        interatomic_electrostatic_potentials = _get_coulomb_potentials(all_atoms, interatomic_distances)
        interatomic_vanderwaals_potentials = _get_lennard_jones_potentials(all_atoms, interatomic_distances)

    # assign features
    if isinstance(graph.edges[0].id, AtomicContact):
        for edge in graph.edges:        
            ## find the indices
            contact = edge.id
            atom1_index = atom_dict[contact.atom1]
            atom2_index = atom_dict[contact.atom2]
            ## set features
            edge.features[Efeat.SAMERES] = float( contact.atom1.residue == contact.atom2.residue)  # 1.0 for True; 0.0 for False
            edge.features[Efeat.SAMECHAIN] = float( contact.atom1.residue.chain == contact.atom1.residue.chain )  # 1.0 for True; 0.0 for False
            edge.features[Efeat.DISTANCE] = interatomic_distances[atom1_index, atom2_index]
            edge.features[Efeat.COVALENT] = float( edge.features[Efeat.DISTANCE] < MAX_COVALENT_DISTANCE )  # 1.0 for True; 0.0 for False
            edge.features[Efeat.ELECTROSTATIC] = interatomic_electrostatic_potentials[atom1_index, atom2_index]
            if edge.features[Efeat.SAMERES]:
                edge.features[Efeat.VANDERWAALS] = interatomic_vanderwaals_potentials['intra'][atom1_index, atom2_index]
            else:
                edge.features[Efeat.VANDERWAALS] = interatomic_vanderwaals_potentials['inter'][atom1_index, atom2_index]
    
    elif isinstance(contact, ResidueContact):
        for edge in graph.edges:        
            ## find the indices
            contact = edge.id
            atom1_indices = [atom_dict[atom] for atom in contact.residue1.atoms]
            atom2_indices = [atom_dict[atom] for atom in contact.residue2.atoms]
            ## set features
            edge.features[Efeat.SAMECHAIN] = float( contact.residue1.chain == contact.residue2.chain )  # 1.0 for True; 0.0 for False
            edge.features[Efeat.DISTANCE] = np.min([[interatomic_distances[a1, a2] for a1 in atom1_indices] for a2 in atom2_indices])
            edge.features[Efeat.COVALENT] = float( edge.features[Efeat.DISTANCE] < MAX_COVALENT_DISTANCE )  # 1.0 for True; 0.0 for False
            edge.features[Efeat.ELECTROSTATIC] = np.sum([[interatomic_electrostatic_potentials[a1, a2] for a1 in atom1_indices] for a2 in atom2_indices])
            edge.features[Efeat.VANDERWAALS] = np.sum([[interatomic_vanderwaals_potentials['inter'][a1, a2] for a1 in atom1_indices] for a2 in atom2_indices])
