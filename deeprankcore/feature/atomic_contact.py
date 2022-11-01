from typing import List
import logging
import numpy as np
from scipy.spatial import distance_matrix
from deeprankcore.models.structure import Atom
from deeprankcore.models.graph import Graph, Edge
from deeprankcore.models.contact import ResidueContact, AtomicContact
from deeprankcore.domain.features import edgefeats as Efeat
from deeprankcore.domain.forcefield import atomic_forcefield, COULOMB_CONSTANT, EPSILON0, MAX_COVALENT_DISTANCE
from deeprankcore.models.forcefield.vanderwaals import VanderwaalsParam
from deeprankcore.models.error import UnknownAtomError


_log = logging.getLogger(__name__)


def get_coulomb_potentials(distances: np.ndarray, charges: List[float]) -> np.ndarray:
    """ Calculate the Coulomb potentials, given a distance matrix and a list of charges of equal size.

        Warning: there's no distance cutoff here. The radius of influence is assumed to infinite
    """

    # check for the correct matrix shape
    charge_count = len(charges)
    if charge_count != distances.shape[0] or charge_count != distances.shape[1]:
        raise ValueError("Cannot calculate potentials between {} charges and {} distances" # pylint: disable=consider-using-f-string
                         .format(charge_count, "x".join([str(d) for d in distances.shape])))

    # calculate the potentials
    potentials = np.expand_dims(charges, axis=0) * np.expand_dims(charges, axis=1) \
                 * COULOMB_CONSTANT / (EPSILON0 * distances)

    return potentials


def get_coulomb_potentials_new(atoms1: List[Atom], atoms2: List[Atom]) -> np.ndarray:
    # calculate distances
    positions1 = [atom.position for atom in atoms1]
    positions2 = [atom.position for atom in atoms2]
    distances = distance_matrix(positions1, positions2)

    # find charges
    charges1 = [atomic_forcefield.get_charge(atom) for atom in atoms1]
    charges2 = [atomic_forcefield.get_charge(atom) for atom in atoms2]

    # calculate potentials
    coulomb_potentials = np.expand_dims(charges1, axis=1) * np.expand_dims(charges2, axis=0) * COULOMB_CONSTANT / (EPSILON0 * distances)

    return coulomb_potentials


def get_lennard_jones_potentials_new(atoms1: List[Atom], atoms2: List[Atom]) -> np.ndarray:
    # calculate distances
    positions1 = [atom.position for atom in atoms1]
    positions2 = [atom.position for atom in atoms2]
    distances = distance_matrix(positions1, positions2)

    # calculate vanderwaals potentials
    sigmas1 = [atomic_forcefield.get_vanderwaals_parameters(atom).inter_sigma for atom in atoms1]
    sigmas2 = [atomic_forcefield.get_vanderwaals_parameters(atom).inter_sigma for atom in atoms2]       
    mean_sigmas = (np.array(sigmas1).reshape(-1, 1) + sigmas2) / 2

    eps1 = [atomic_forcefield.get_vanderwaals_parameters(atom).inter_epsilon for atom in atoms1]
    eps2 = [atomic_forcefield.get_vanderwaals_parameters(atom).inter_epsilon for atom in atoms2]
    geomean_eps = np.sqrt((np.array(eps1).reshape(-1, 1) * eps2)) # sqrt(eps1*eps2)
    
    lennard_jones_potentials = 4.0 * geomean_eps * ((mean_sigmas / distances) ** 12 - (mean_sigmas / distances) ** 6)

    return lennard_jones_potentials



def get_lennard_jones_potentials(distances: np.ndarray, atoms: List[Atom],
                                 vanderwaals_parameters: List[VanderwaalsParam]) -> np.ndarray:
    """ Calculate Lennard-Jones potentials, given a distance matrix and a list of atoms with vanderwaals parameters of equal size.

         Warning: there's no distance cutoff here. The radius of influence is assumed to infinite
    """

    # check for the correct data shapes
    atom_count = len(atoms)
    if atom_count != len(vanderwaals_parameters):
        raise ValueError(f"The number of atoms ({atom_count}) does not match the number of vanderwaals parameters ({len(vanderwaals_parameters)})")
    if atom_count != distances.shape[0] or atom_count != distances.shape[1]:
        raise ValueError("Cannot calculate potentials between {} atoms and {} distances" # pylint: disable=consider-using-f-string
                         .format(atom_count, "x".join([str(d) for d in distances.shape])))

    # collect parameters
    sigmas1 = np.empty((atom_count, atom_count))
    sigmas2 = np.empty((atom_count, atom_count))
    epsilons1 = np.empty((atom_count, atom_count))
    epsilons2 = np.empty((atom_count, atom_count))
    for atom1_index in range(atom_count):
        for atom2_index in range(atom_count):
            atom1 = atoms[atom1_index]
            atom2 = atoms[atom2_index]

            # Which parameter we take, depends on whether the contact is intra- or intermolecular.
            if atom1.residue != atom2.residue:

                sigmas1[atom1_index][atom2_index] = vanderwaals_parameters[atom1_index].inter_sigma
                sigmas2[atom1_index][atom2_index] = vanderwaals_parameters[atom2_index].inter_sigma

                epsilons1[atom1_index][atom2_index] = vanderwaals_parameters[atom1_index].inter_epsilon
                epsilons2[atom1_index][atom2_index] = vanderwaals_parameters[atom2_index].inter_epsilon
            else:
                sigmas1[atom1_index][atom2_index] = vanderwaals_parameters[atom1_index].intra_sigma
                sigmas2[atom1_index][atom2_index] = vanderwaals_parameters[atom2_index].intra_sigma

                epsilons1[atom1_index][atom2_index] = vanderwaals_parameters[atom1_index].intra_epsilon
                epsilons2[atom1_index][atom2_index] = vanderwaals_parameters[atom2_index].intra_epsilon

    # calculate potentials
    sigmas = 0.5 * (sigmas1 + sigmas2)
    epsilons = np.sqrt(epsilons1 * epsilons2)
    potentials = 4.0 * epsilons * ((sigmas / distances) ** 12 - (sigmas / distances) ** 6)

    return potentials


def add_features_for_atoms(edges: List[Edge]): # pylint: disable=too-many-locals

    # get a set of all the atoms involved
    atoms = set([])
    for edge in edges:
        contact = edge.id
        atoms.add(contact.atom1)
        atoms.add(contact.atom2)
    atoms = list(atoms)

    # get all atomic parameters
    atom_indices = {}
    positions = []
    atom_charges = []
    atom_vanderwaals_parameters = []
    atom_chains = []
    atom_residues = []
    for atom_index, atom in enumerate(atoms):
        try:
            charge = atomic_forcefield.get_charge(atom)
            vanderwaals = atomic_forcefield.get_vanderwaals_parameters(atom)

        except UnknownAtomError:
            _log.warning("Ignoring atom %s, because it's unknown to the forcefield", atom)

            # set parameters to zero, so that the potential becomes zero
            charge = 0.0
            vanderwaals = VanderwaalsParam(0.0, 0.0, 0.0, 0.0)

        atom_charges.append(charge)
        atom_vanderwaals_parameters.append(vanderwaals)
        positions.append(atom.position)
        atom_indices[atom] = atom_index
        atom_chains.append(atom.residue.chain.id)
        atom_residues.append(atom.residue.number)

    # calculate the distance matrix for those atoms
    interatomic_distances = distance_matrix(positions, positions, p=2)

    # calculate potentials
    interatomic_electrostatic_potentials = get_coulomb_potentials(interatomic_distances, atom_charges)
    interatomic_vanderwaals_potentials = get_lennard_jones_potentials(interatomic_distances, atoms, atom_vanderwaals_parameters)

    # determine which atoms are close enough to form a covalent bond
    covalent_neighbours = interatomic_distances < MAX_COVALENT_DISTANCE

    # set the edge features
    for _, edge in enumerate(edges):
        contact = edge.id
        atom1_index = atom_indices[contact.atom1]
        atom2_index = atom_indices[contact.atom2]

        edge.features[Efeat.DISTANCE] = interatomic_distances[atom1_index, atom2_index]
        edge.features[Efeat.VANDERWAALS] = interatomic_vanderwaals_potentials[atom1_index, atom2_index]
        edge.features[Efeat.ELECTROSTATIC] = interatomic_electrostatic_potentials[atom1_index, atom2_index]

        if covalent_neighbours[atom1_index, atom2_index]:
            edge.features[Efeat.COVALENT] = 1.0
        else:
            edge.features[Efeat.COVALENT] = 0.0
        
        if atom_chains[atom1_index] == atom_chains[atom2_index]:
            edge.features[Efeat.SAMECHAIN] = 1.0
            if atom_residues[atom1_index] == atom_residues[atom2_index]:
                edge.features[Efeat.SAMERES] = 1.0
            else:
                edge.features[Efeat.SAMERES] = 0.0
        else:
            edge.features[Efeat.SAMECHAIN] = 0.0
            edge.features[Efeat.SAMERES] = 0.0

def add_features_for_residues(edges: List[Edge]): # pylint: disable=too-many-locals
    # set the edge features
    for edge in edges:
        contact = edge.id

        atoms1 = contact.residue1.atoms
        atoms2 = contact.residue2.atoms
        
        # calculate distances
        positions1 = [atom.position for atom in atoms1]
        positions2 = [atom.position for atom in atoms2]
        distances = distance_matrix(positions1, positions2)

        # calculate electrostatic potentials
        charges1 = [atomic_forcefield.get_charge(x) for x in atoms1]
        charges2 = [atomic_forcefield.get_charge(x) for x in atoms2]
        coulomb_potentials = np.expand_dims(charges1, axis=1) * np.expand_dims(charges2, axis=0) * COULOMB_CONSTANT / (EPSILON0 * distances)

        # calculate vanderwaals potentials
        sigmas1 = [atomic_forcefield.get_vanderwaals_parameters(x).inter_sigma for x in atoms1]
        sigmas2 = [atomic_forcefield.get_vanderwaals_parameters(x).inter_sigma for x in atoms2]       
        mean_sigmas = (np.array(sigmas1).reshape(-1, 1) + sigmas2) / 2
 
        eps1 = [atomic_forcefield.get_vanderwaals_parameters(x).inter_epsilon for x in atoms1]
        eps2 = [atomic_forcefield.get_vanderwaals_parameters(x).inter_epsilon for x in atoms2]
        geomean_eps = np.sqrt((np.array(eps1).reshape(-1, 1) * eps2))
        
        lennard_jones_potential = 4.0 * geomean_eps * ((mean_sigmas / distances) ** 12 - (mean_sigmas / distances) ** 6)

        # set features
        edge.features[Efeat.SAMECHAIN] = float( contact.residue1.chain == contact.residue2.chain ) # 1.0 for True; 0.0 for False
        edge.features[Efeat.DISTANCE] = np.min(distances) # minimum atom distance is considered as the distance between 2 residues
        edge.features[Efeat.COVALENT] = float( edge.features[Efeat.DISTANCE] < MAX_COVALENT_DISTANCE ) # 1.0 for True; 0.0 for False
        edge.features[Efeat.ELECTROSTATIC] = np.sum(coulomb_potentials)
        edge.features[Efeat.VANDERWAALS] = np.sum(lennard_jones_potential)


def add_features(pdb_path: str, graph: Graph, *args, **kwargs): # pylint: disable=unused-argument

    if isinstance(graph.edges[0].id, ResidueContact):
        add_features_for_residues(graph.edges)

    elif isinstance(graph.edges[0].id, AtomicContact):
        add_features_for_atoms(graph.edges)
    else:
        raise TypeError(f"Unexpected edge type: {type(graph.edges[0].id)}")