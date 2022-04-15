from typing import Dict, List

import numpy
from scipy.spatial import distance_matrix

from deeprank_gnn.models.structure import Structure, Atom
from deeprank_gnn.models.graph import Edge
from deeprank_gnn.feature.edge import EdgeFeatureProvider
from deeprank_gnn.models.contact import ResidueContact, AtomicContact
from deeprank_gnn.domain.feature import FEATURENAME_EDGEDISTANCE, FEATURENAME_EDGEVANDERWAALS, FEATURENAME_EDGECOULOMB
from deeprank_gnn.domain.forcefield import atomic_forcefield, COULOMB_CONSTANT, EPSILON0
from deeprank_gnn.models.forcefield.vanderwaals import VanderwaalsParam


def get_coulomb_potentials(distances: numpy.ndarray, charges: List[float]) -> numpy.ndarray:
    """ Calculate the Coulomb potentials, given a distance matrix and a list of charges of equal size.

        Warning: there's no distance cutoff here. The radius of influence is assumed to infinite
    """

    # check for the correct matrix shape
    charge_count = len(charges)
    if charge_count != distances.shape[0] or charge_count != distances.shape[1]:
        raise ValueError("Cannot calculate distances between {} charges and {} distances"
                         .format(charge_count, "x".join(distances.shape)))

    # calculate the potentials
    potentials = numpy.expand_dims(charges, axis=0) * numpy.expand_dims(charges, axis=1) \
                 * COULOMB_CONSTANT / (EPSILON0 * distances)

    return potentials


def get_lennard_jones_potentials(distances: numpy.ndarray, atoms: List[Atom],
                                 vanderwaals_parameters: List[VanderwaalsParam]) -> numpy.ndarray:
    """ Calculate Lennard-Jones potentials, given a distance matrix and a list of atoms with vanderwaals parameters of equal size.

         Warning: there's no distance cutoff here. The radius of influence is assumed to infinite
    """

    # check for the correct data shapes
    atom_count = len(atoms)
    if atom_count != len(vanderwaals_parameters):
        raise ValueError("The number of atoms ({}) does not match the number of vanderwaals parameters ({})"
                         .format(atom_count, len(vanderwaals_parameters)))
    if atom_count != distances.shape[0] or atom_count != distances.shape[1]:
        raise ValueError("Cannot calculate distances between {} atoms and {} distances"
                         .format(atom_count, "x".join(distances.shape)))

    # collect parameters
    sigmas1 = numpy.empty((atom_count, atom_count))
    sigmas2 = numpy.empty((atom_count, atom_count))
    epsilons1 = numpy.empty((atom_count, atom_count))
    epsilons2 = numpy.empty((atom_count, atom_count))
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
    epsilons = numpy.sqrt(sigmas1 * sigmas2)
    potentials = 4.0 * epsilons * ((sigmas / distances) ** 12 - (sigmas / distances) ** 6)

    return potentials


def add_features_for_atoms(edges: List[Edge]):

    # get a set of all the atoms involved
    atoms = set([])
    for edge in edges:
        contact = edge.id
        atoms.add(contact.atom1)
        atoms.add(contact.atom2)
    atoms = list(atoms)

    # get the positions of those atoms (and map atoms back to their index for quick lookup)
    atom_indices = {}
    positions = numpy.empty((len(atoms), 3))
    for atom_index, atom in enumerate(atoms):
        positions[atom_index] = atom.position
        atom_indices[atom] = atom_index

    # calculate the distance matrix for those atoms
    interatomic_distances = distance_matrix(positions, positions, p=2)

    # get forcefield parameters
    atom_charges = [atomic_forcefield.get_charge(atom) for atom in atoms]
    atom_vanderwaals_parameters = [atomic_forcefield.get_vanderwaals_parameters(atom) for atom in atoms]

    # calculate potentials
    interatomic_electrostatic_potentials = get_coulomb_potentials(interatomic_distances, atom_charges)
    interatomic_vanderwaals_potentials = get_lennard_jones_potentials(interatomic_distances, atoms, atom_vanderwaals_parameters)

    # set the features
    for edge_index, edge in enumerate(edges):
        contact = edge.id
        atom1_index = atom_indices[contact.atom1]
        atom2_index = atom_indices[contact.atom2]

        edge.features[FEATURENAME_EDGEDISTANCE] = interatomic_distances[atom1_index, atom2_index]
        edge.features[FEATURENAME_EDGEVANDERWAALS] = interatomic_vanderwaals_potentials[atom1_index, atom2_index]
        edge.features[FEATURENAME_EDGECOULOMB] = interatomic_electrostatic_potentials[atom1_index, atom2_index]


def add_features_for_residues(edges: List[Edge]):
    # get a set of all the atoms involved
    atoms = set([])
    for edge in edges:
        contact = edge.id
        atoms |= contact.residue1.atoms
        atoms |= contact.residue2.atoms
    atoms = list(atoms)

    # get the positions of those atoms (and map atoms back to their index for quick lookup)
    atom_indices = {}
    positions = numpy.empty((len(atoms), 3))
    for atom_index, atom in enumerate(atoms):
        positions[atom_index] = atom.position
        atom_indices[atom] = atom_index

    # calculate the distance matrix for those atoms
    interatomic_distances = distance_matrix(positions, positions, p=2)

    # get forcefield parameters
    atom_charges = [atomic_forcefield.get_charge(atom) for atom in atoms]
    atom_vanderwaals_parameters = [atomic_forcefield.get_vanderwaals_parameters(atom) for atom in atoms]

    # calculate potentials
    interatomic_electrostatic_potentials = get_coulomb_potentials(interatomic_distances, atom_charges)
    interatomic_vanderwaals_potentials = get_lennard_jones_potentials(interatomic_distances, atoms, atom_vanderwaals_parameters)

    # set the features
    for edge_index, edge in enumerate(edges):
        contact = edge.id
        for atom1 in contact.residue1.atoms:
            for atom2 in contact.residue2.atoms:
                atom1_index = atom_indices[atom1]
                atom2_index = atom_indices[atom2]

                edge.features[FEATURENAME_EDGEDISTANCE] = min(edge.features.get(FEATURENAME_EDGEDISTANCE, 0.0),
                                                              interatomic_distances[atom1_index, atom2_index])

                edge.features[FEATURENAME_EDGEVANDERWAALS] = (edge.features.get(FEATURENAME_EDGEVANDERWAALS, 0.0) +
                                                              interatomic_vanderwaals_potentials)

                edge.features[FEATURENAME_EDGECOULOMB] = (edge.features.get(FEATURENAME_EDGECOULOMB, 0.0) +
                                                          interatomic_electrostatic_potentials)


def add_features(pdb_path: str, edges: List[Edge]) -> Dict[str, numpy.ndarray]:

    if type(edges[0].id) == ResidueContact:
        return add_features_for_residues(edges)

    elif type(edges[0].id) == AtomicContact:
        return add_features_for_atoms(edges)
    else:
        raise TypeError(type(edges[0].id))




