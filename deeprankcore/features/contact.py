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
import numpy.typing as npt

_log = logging.getLogger(__name__)

# cutoff distances for 1-3 and 1-4 pairing. See issue: https://github.com/DeepRank/deeprank-core/issues/357#issuecomment-1461813723
covalent_cutoff = 2.1
cutoff_13 = 3.6
cutoff_14 = 4.2


def _get_electrostatic_energy(atoms: List[Atom], distances: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculates all pairwise electrostatic potential energies (Coulomb potentials) between all atoms in the structure.

    Warning: there's no distance cutoff here. The radius of influence is assumed to infinite.
    However, the potential tends to 0 at large distance.

    Args:
        atoms (List[Atom]): list of all atoms in the structure 
        distances (npt.NDArray[np.float64]): matrix of pairwise distances between all atoms in the structure 
            in the format that is the output of scipy.spatial's distance_matrix (i.e. a diagonally symmetric matrix)

    Returns:
        npt.NDArray[np.float64]: matrix containing all pairwise electrostatic potential energies in same format as `distances`
    """

    EPSILON0 = 1.0
    COULOMB_CONSTANT = 332.0636
    charges = [atomic_forcefield.get_charge(atom) for atom in atoms]
    electrostatic_energy = np.expand_dims(charges, axis=1) * np.expand_dims(charges, axis=0) * COULOMB_CONSTANT / (EPSILON0 * distances)
    electrostatic_energy[distances < cutoff_13] = 0
    return electrostatic_energy


def _get_vdw_energy(atoms: List[Atom], distances: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculates all pairwise Van der Waals potential energies (Lennard-Jones potentials) between all atoms in the structure.

    Warning: there's no distance cutoff here. The radius of influence is assumed to infinite.
    However, the potential tends to 0 at large distance.
    
    Args:
        atoms (List[Atom]): list of all atoms in the structure 
        distances (npt.NDArray[np.float64]): matrix of pairwise distances between all atoms in the structure 
            in the format that is the output of scipy.spatial's distance_matrix (i.e. a diagonally symmetric matrix)

    Returns:
        npt.NDArray[np.float64]: matrix containing all pairwise Van der Waals potential energies in same format as `distances`
    """

    # calculate main vdw energies
    sigmas = [atomic_forcefield.get_vanderwaals_parameters(atom).sigma_main for atom in atoms]
    epsilons = [atomic_forcefield.get_vanderwaals_parameters(atom).epsilon_main for atom in atoms]
    mean_sigmas = 0.5 * np.add.outer(sigmas,sigmas)
    geomean_eps = np.sqrt(np.multiply.outer(epsilons,epsilons))     # sqrt(eps1*eps2)
    vdw_energy = 4.0 * geomean_eps * ((mean_sigmas / distances) ** 12 - (mean_sigmas / distances) ** 6)

    # calculate energies for 1-4 pairs
    sigmas = [atomic_forcefield.get_vanderwaals_parameters(atom).sigma_14 for atom in atoms]
    epsilons = [atomic_forcefield.get_vanderwaals_parameters(atom).epsilon_14 for atom in atoms]
    mean_sigmas = 0.5 * np.add.outer(sigmas,sigmas)
    geomean_eps = np.sqrt(np.multiply.outer(epsilons,epsilons))     # sqrt(eps1*eps2)
    energy_14pairs = 4.0 * geomean_eps * ((mean_sigmas / distances) ** 12 - (mean_sigmas / distances) ** 6)

    # adjust vdw energy for close contacts
    vdw_energy[distances < cutoff_14] = energy_14pairs[distances < cutoff_14]
    vdw_energy[distances < cutoff_13] = 0
    return vdw_energy


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
        interatomic_electrostatic_energy = _get_electrostatic_energy(all_atoms, interatomic_distances)
        interatomic_vanderwaals_energy = _get_vdw_energy(all_atoms, interatomic_distances)

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
            edge.features[Efeat.COVALENT] = float( edge.features[Efeat.DISTANCE] < covalent_cutoff )  # 1.0 for True; 0.0 for False
            edge.features[Efeat.ELECTROSTATIC] = interatomic_electrostatic_energy[atom1_index, atom2_index]
            edge.features[Efeat.VANDERWAALS] = interatomic_vanderwaals_energy[atom1_index, atom2_index]
    
    elif isinstance(contact, ResidueContact):
        for edge in graph.edges:        
            ## find the indices
            contact = edge.id
            atom1_indices = [atom_dict[atom] for atom in contact.residue1.atoms]
            atom2_indices = [atom_dict[atom] for atom in contact.residue2.atoms]
            ## set features
            edge.features[Efeat.SAMECHAIN] = float( contact.residue1.chain == contact.residue2.chain )  # 1.0 for True; 0.0 for False
            edge.features[Efeat.DISTANCE] = np.min([[interatomic_distances[a1, a2] for a1 in atom1_indices] for a2 in atom2_indices])
            edge.features[Efeat.COVALENT] = float( edge.features[Efeat.DISTANCE] < covalent_cutoff )  # 1.0 for True; 0.0 for False
            edge.features[Efeat.ELECTROSTATIC] = np.sum([[interatomic_electrostatic_energy[a1, a2] for a1 in atom1_indices] for a2 in atom2_indices])
            edge.features[Efeat.VANDERWAALS] = np.sum([[interatomic_vanderwaals_energy[a1, a2] for a1 in atom1_indices] for a2 in atom2_indices])
