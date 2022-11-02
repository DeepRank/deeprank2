from typing import List
import logging
import numpy as np
from scipy.spatial import distance_matrix
from deeprankcore.models.structure import Atom
from deeprankcore.models.graph import Graph
from deeprankcore.models.contact import ResidueContact, AtomicContact
from deeprankcore.domain.features import edgefeats as Efeat
from deeprankcore.domain.forcefield import atomic_forcefield, COULOMB_CONSTANT, EPSILON0, MAX_COVALENT_DISTANCE

_log = logging.getLogger(__name__)


def _get_coulomb_potentials(atoms1: List[Atom], atoms2: List[Atom], distances: np.ndarray) -> np.ndarray:
    """ 
        Calculate pairwise Coulomb potentials between each Atom from atoms1 and each Atom from atoms2.
        Warning: there's no distance cutoff here. The radius of influence is assumed to infinite (but the potential tends to 0 at large distance)
    """

    # find charges
    charges1 = [atomic_forcefield.get_charge(atom) for atom in atoms1]
    charges2 = [atomic_forcefield.get_charge(atom) for atom in atoms2]

    # calculate potentials
    coulomb_potentials = np.expand_dims(charges1, axis=1) * np.expand_dims(charges2, axis=0) * COULOMB_CONSTANT / (EPSILON0 * distances)

    return coulomb_potentials


def _get_lennard_jones_potentials(atoms1: List[Atom], atoms2: List[Atom], distances: np.ndarray) -> np.ndarray:
    """ 
        Calculate Lennard-Jones potentials between each Atom from atoms1 and each Atom from atoms2.
        Warning: there's no distance cutoff here. The radius of influence is assumed to infinite (but the potential tends to 0 at large distance)
    """

    # calculate vanderwaals potentials
    if atoms1[0].residue == atoms2[0].residue: # use intra- parameters
        sigmas1 = [atomic_forcefield.get_vanderwaals_parameters(atom).intra_sigma for atom in atoms1]
        sigmas2 = [atomic_forcefield.get_vanderwaals_parameters(atom).intra_sigma for atom in atoms2]       
        epsilon1 = [atomic_forcefield.get_vanderwaals_parameters(atom).intra_epsilon for atom in atoms1]
        epsilon2 = [atomic_forcefield.get_vanderwaals_parameters(atom).intra_epsilon for atom in atoms2]
    else: # use inter- parameters
        sigmas1 = [atomic_forcefield.get_vanderwaals_parameters(atom).inter_sigma for atom in atoms1]
        sigmas2 = [atomic_forcefield.get_vanderwaals_parameters(atom).inter_sigma for atom in atoms2]       
        epsilon1 = [atomic_forcefield.get_vanderwaals_parameters(atom).inter_epsilon for atom in atoms1]
        epsilon2 = [atomic_forcefield.get_vanderwaals_parameters(atom).inter_epsilon for atom in atoms2]
        
    mean_sigmas = 0.5 * np.add.outer(sigmas1,sigmas2)
    geomean_eps = np.sqrt(np.multiply.outer(epsilon1,epsilon2)) # sqrt(eps1*eps2)
    lennard_jones_potentials = 4.0 * geomean_eps * ((mean_sigmas / distances) ** 12 - (mean_sigmas / distances) ** 6)

    return lennard_jones_potentials


def add_features(pdb_path: str, graph: Graph, *args, **kwargs): # pylint: disable=unused-argument
    for edge in graph.edges:
        contact = edge.id

        if isinstance(contact, ResidueContact):
            atoms1 = contact.residue1.atoms
            atoms2 = contact.residue2.atoms
        elif isinstance(contact, AtomicContact):
            atoms1 = [contact.atom1]
            atoms2 = [contact.atom2]
            edge.features[Efeat.SAMERES] = float( contact.atom1.residue == contact.atom2.residue) # 1.0 for True; 0.0 for False
        else:
            raise TypeError(
                f"Unexpected edge type: {type(contact)} for {edge}")

        # calculate the distances matrix
        interatomic_distances = distance_matrix([atom.position for atom in atoms1], 
                                                [atom.position for atom in atoms2])

        # calculate potentials matrices
        interatomic_electrostatic_potentials = _get_coulomb_potentials(atoms1, atoms2, interatomic_distances)
        interatomic_vanderwaals_potentials = _get_lennard_jones_potentials(atoms1, atoms2, interatomic_distances)

        # set features
        edge.features[Efeat.DISTANCE] = np.min(interatomic_distances) # minimum atom distance is considered as the distance between 2 residues
        edge.features[Efeat.ELECTROSTATIC] = np.sum(interatomic_electrostatic_potentials)
        edge.features[Efeat.VANDERWAALS] = np.sum(interatomic_vanderwaals_potentials)
        edge.features[Efeat.SAMECHAIN] = float( atoms1[0].residue.chain == atoms2[0].residue.chain ) # 1.0 for True; 0.0 for False
        edge.features[Efeat.COVALENT] = float( edge.features[Efeat.DISTANCE] < MAX_COVALENT_DISTANCE ) # 1.0 for True; 0.0 for False
