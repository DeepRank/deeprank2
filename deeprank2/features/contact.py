import logging
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance_matrix

from deeprank2.domain import edgestorage as Efeat
from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.pair import AtomicContact, ResidueContact
from deeprank2.molstruct.residue import SingleResidueVariant
from deeprank2.utils.graph import Graph
from deeprank2.utils.parsing import atomic_forcefield

_log = logging.getLogger(__name__)

# for cutoff distances, see: https://github.com/DeepRank/deeprank2/issues/357#issuecomment-1461813723
covalent_cutoff = 2.1
cutoff_13 = 3.6
cutoff_14 = 4.2
EPSILON0 = 1.0
COULOMB_CONSTANT = 332.0636


def _get_nonbonded_energy(
    atoms: list[Atom],
    distances: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculates all pairwise electrostatic (Coulomb) and Van der Waals (Lennard Jones) potential energies between all atoms in the structure.

    Warning: there's no distance cutoff here. The radius of influence is assumed to infinite.
    However, the potential tends to 0 at large distance.

    Args:
        atoms: list of all atoms in the structure
        distances: matrix of pairwise distances between all atoms in the structure
            in the format that is the output of scipy.spatial's distance_matrix (i.e. a diagonally symmetric matrix)

    Returns:
        Tuple [NDArray[np.float64], NDArray[np.float64]]: matrices in same format as `distances` containing
            all pairwise electrostatic potential energies and all pairwise Van der Waals potential energies
    """
    # ELECTROSTATIC POTENTIAL
    charges = [atomic_forcefield.get_charge(atom) for atom in atoms]
    E_elec = np.expand_dims(charges, axis=1) * np.expand_dims(charges, axis=0) * COULOMB_CONSTANT / (EPSILON0 * distances)

    # VAN DER WAALS POTENTIAL
    # calculate main vdw energies
    sigmas = [atomic_forcefield.get_vanderwaals_parameters(atom).sigma_main for atom in atoms]
    epsilons = [atomic_forcefield.get_vanderwaals_parameters(atom).epsilon_main for atom in atoms]
    mean_sigmas = 0.5 * np.add.outer(sigmas, sigmas)
    geomean_eps = np.sqrt(np.multiply.outer(epsilons, epsilons))  # sqrt(eps1*eps2)
    E_vdw = 4.0 * geomean_eps * ((mean_sigmas / distances) ** 12 - (mean_sigmas / distances) ** 6)

    # calculate vdw energies for 1-4 pairs
    sigmas = [atomic_forcefield.get_vanderwaals_parameters(atom).sigma_14 for atom in atoms]
    epsilons = [atomic_forcefield.get_vanderwaals_parameters(atom).epsilon_14 for atom in atoms]
    mean_sigmas = 0.5 * np.add.outer(sigmas, sigmas)
    geomean_eps = np.sqrt(np.multiply.outer(epsilons, epsilons))  # sqrt(eps1*eps2)
    E_vdw_14pairs = 4.0 * geomean_eps * ((mean_sigmas / distances) ** 12 - (mean_sigmas / distances) ** 6)

    # Fix energies for close contacts on same chain
    chains = [atom.residue.chain.id for atom in atoms]
    chain_matrix = [[chain_1 == chain_2 for chain_2 in chains] for chain_1 in chains]
    pair_14 = np.logical_and(distances < cutoff_14, chain_matrix)
    pair_13 = np.logical_and(distances < cutoff_13, chain_matrix)

    E_vdw[pair_14] = E_vdw_14pairs[pair_14]
    E_vdw[pair_13] = 0
    E_elec[pair_13] = 0

    return E_elec, E_vdw


def add_features(  # noqa:D103
    pdb_path: str,  # noqa: ARG001
    graph: Graph,
    single_amino_acid_variant: SingleResidueVariant | None = None,  # noqa: ARG001
) -> None:
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
            for atom in contact.residue1.atoms + contact.residue2.atoms:
                all_atoms.add(atom)
    else:
        msg = f"Unexpected edge type: {type(graph.edges[0].id)}"
        raise TypeError(msg)

    all_atoms = list(all_atoms)
    atom_dict = {atom: i for i, atom in enumerate(all_atoms)}

    # make pairwise calculations between all atoms in the set
    with warnings.catch_warnings(record=RuntimeWarning):
        warnings.simplefilter("ignore")
        positions = [atom.position for atom in all_atoms]
        interatomic_distances = distance_matrix(positions, positions)
        (
            interatomic_electrostatic_energy,
            interatomic_vanderwaals_energy,
        ) = _get_nonbonded_energy(all_atoms, interatomic_distances)

    # assign features
    for edge in graph.edges:
        contact = edge.id

        if isinstance(contact, AtomicContact):
            ## find the indices
            atom1_index = atom_dict[contact.atom1]
            atom2_index = atom_dict[contact.atom2]
            ## set features
            edge.features[Efeat.SAMERES] = float(contact.atom1.residue == contact.atom2.residue)
            edge.features[Efeat.SAMECHAIN] = float(contact.atom1.residue.chain == contact.atom1.residue.chain)
            edge.features[Efeat.DISTANCE] = interatomic_distances[atom1_index, atom2_index]
            edge.features[Efeat.ELEC] = interatomic_electrostatic_energy[atom1_index, atom2_index]
            edge.features[Efeat.VDW] = interatomic_vanderwaals_energy[atom1_index, atom2_index]

        elif isinstance(contact, ResidueContact):
            ## find the indices
            atom1_indices = [atom_dict[atom] for atom in contact.residue1.atoms]
            atom2_indices = [atom_dict[atom] for atom in contact.residue2.atoms]
            ## set features
            edge.features[Efeat.SAMECHAIN] = float(contact.residue1.chain == contact.residue2.chain)
            edge.features[Efeat.DISTANCE] = np.min([[interatomic_distances[a1, a2] for a1 in atom1_indices] for a2 in atom2_indices])
            edge.features[Efeat.ELEC] = np.sum([[interatomic_electrostatic_energy[a1, a2] for a1 in atom1_indices] for a2 in atom2_indices])
            edge.features[Efeat.VDW] = np.sum([[interatomic_vanderwaals_energy[a1, a2] for a1 in atom1_indices] for a2 in atom2_indices])

        # Calculate irrespective of node type
        edge.features[Efeat.COVALENT] = float(edge.features[Efeat.DISTANCE] < covalent_cutoff and edge.features[Efeat.SAMECHAIN])
