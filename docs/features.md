# Features

Features implemented in the code-base are defined in `deeprankcore.feature` subpackage.


## Custom features

Users can add custom features by creating a new module and placing it in `deeprankcore.feature` subpackage. One requirement for any feature module is to implement an `add_features` function, as shown below. This will be used in `deeprankcore.models.query` to add the features to the nodes or edges of the graph.

```python
from typing import Optional

from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.utils.graph import Graph


def add_features(
    pdb_path: str, graph: Graph,
    single_amino_acid_variant: Optional[SingleResidueVariant] = None
    ):
    pass
```

The following is a brief description of the features already implemented in the code-base, for each features' module. 

## Default node features 
For atomic graphs, when features relate to residues then _all_ atoms of one residue receive the feature value for that residue.

### Core properties of atoms and residues: `deeprankcore.features.components`
These features relate to the chemical components (atoms and amino acid residues) of which the graph is composed. Detailed information and descrepancies between sources are described can be found in `deeprankcore.domain.aminoacidlist.py`.

#### Atom properties:
These features are only used in atomic graphs.

- `atom_type`: One hot encoding of the atomic element. Options are: C, O, N, S, P, H.
- `atom_charge`: Atomic charge in Coulomb (float). Taken from `deeprankcore.domain.forcefield.patch.top`.
- `pdb_occupancy`: Proportion of structures where the atom was detected at this position (float). In some cases a single atom was detected at different positions, in which case separate structures exist whose occupancies sum to 1. Only the highest occupancy atom is used by deeprankcore. 

#### Residue properties:
- `res_type`: One-hot encoding of the amino acid residue (size 20).
- `polarity`: One hot encoding of the polarity of the amino acid (options: NONPOLAR, POLAR, NEGATIVE, POSITIVE). Note that sources vary on the polarity for few of the amino acids; see detailed information in `deeprankcore.domain.aminoacidlist.py`.
- `res_size`: The number of non-hydrogen atoms in the side chain (int). 
- `res_mass`: The (average) residue mass in Da (float).
- `res_charge`: The charge of the residue (in fully protonated state) in Coulomb (int). Charge is calculated from summing all atoms in the residue, which results in a charge of 0 for all polar and nonpolar residues, +1 for positive residues and -1 for negative residues.
- `res_pI`: The isolectric point, i.e. the pH at which the molecule has no net electric charge (float).

- `hb_donors`, `hb_acceptors`: The number of hydrogen bond donor/acceptor atoms in the residue (int). Hydrogen bonds are noncovalent intermolecular interactions formed between an hydrogen atom (partially positively charged) bound to a small, highly electronegative atom (O, N, F) with an unshared electron pair.

#### Properties related to variant residues:
These features are only used in SingleResidueVariant queries.

- `variant_res`: One hot encoding of variant amino acid (size 20).
- `diff_charge`, `diff_polarity`, `diff_size`, `diff_mass`, `diff_pI`, `diff_hb_donors`, `diff_hb_acceptors`: Subtraction of the wildtype value of indicated feature from the variant value. For example, if the variant has 4 hb_donors and the wildtype has 5, then `diff_hb_donors == -1`.

### Conservation features: `deeprankcore.features.conservation`
These features relate to the conservation state of individual residues.

- `pssm`: [Position-specific scoring matrix](https://en.wikipedia.org/wiki/Position_weight_matrix) (also known as position weight matrix, PWM) values relative to the residue, is a score of the conservation of the amino acid along all 20 amino acids. 
- `info_content`: Information content is the difference between the given PSSM for an amino acid and a uniform distribution (float).
- `conservation` (only used in SingleResidueVariant queries): Conservation of the wild type amino acid (float). *More details required.*
- `diff_conservation` (only used in SingleResidueVariant queries): Subtraction of wildtype conservation from the variant conservation (float). 

### Protein context features:

#### Surface exposure: `deeprankcore.features.exposure`
These features relate to the exposure of residues to the surface, and are computed using [biopython](https://biopython.org/docs/1.81/api/Bio.PDB.html). Note that these features can only be calculated per residue and not per atom.

- `res_depth`: [Residue depth](https://en.wikipedia.org/wiki/Residue_depth) is the average distance (in Å) of the residue to the closest molecule of bulk water (float). See also [`Bio.PDB.ResidueDepth`](https://biopython.org/docs/1.75/api/Bio.PDB.ResidueDepth.html).
- `hse`: [Half sphere exposure (HSE)](https://en.wikipedia.org/wiki/Half_sphere_exposure) is a protein solvent exposure measure indicating how buried an amino acid residue is in a protein (3 float values, see [Bio.PDB.HSExposure](https://biopython.org/docs/dev/api/Bio.PDB.HSExposure.html#module-Bio.PDB.HSExposure) for details).

#### Surface accessibility: `deeprankcore.features.surfacearea`
These features relate to the surface area of the residue, and are computed using [freesasa](https://freesasa.github.io). Note that these features can only be calculated per residue and not per atom.

- `sasa`: [Solvent-Accessible Surface Area](https://en.wikipedia.org/wiki/Accessible_surface_area) is the surface area (in Å^2) of a biomolecule that is accessible to the solvent (float).
- `bsa`: Buried Surface Area is the surface area (in Å^2) that is buried away from the solvent when two or more proteins or subunits associate to form a complex, i.e. it measures the size of the complex interface (float).

#### Secondary structure: `deeprankcore.features.secondary_structure`

- `sec_struct`: One hot encoding of the [DSSP](https://en.wikipedia.org/wiki/DSSP_(algorithm)) assigned secondary structure of the amino acid, using the three major classes (HELIX, STRAND, COIL). Calculated using [DSSP4](https://github.com/PDB-REDO/dssp).

#### Inter-residue contacts (IRCs): `deeprankcore.features.irc`
These features are only calculated for ProteinProteinInterface queries.

- `irc_total`: The number of residues on the other chain that are within a cutoff distance of 5.5 Å (int).
- `irc_nonpolar_nonpolar`, `irc_nonpolar_polar`, `irc_nonpolar_negative`, `irc_nonpolar_positive`, `irc_polar_polar`, `irc_polar_negative`, `irc_polar_positive`, `irc_negative_negative`, `irc_positive_positive`, `irc_negative_positive`: As above, but for specific residue polarity pairings.


## Default edge features 

### Contact features: `deeprankcore.features.contact`
These features relate to relationships between individual nodes.
For atomic graphs, when features relate to residues then _all_ atoms of one residue receive the feature value for that residue.

#### Distance:
- `distance`: Interatomic distance between atoms in Å, computed from the xyz atomic coordinates taken from the .pdb file (float). For residue graphs, the the minimum distance between any atom of each residues is used. 

#### Structure:
These features relate to the structural relationship between nodes.
- `same_chain`: Boolean indicating whether the edge connects nodes belonging to the same chain (1) or separate chains (0).
- `same_res`: Boolean indicating whether atoms belong to the same residue (1) or separate residues (0). Only used in atomic graphs.
- `covalent`: Boolean indicating whether nodes are covalently bound (1) or not (0). Note that covalency is not directly assessed, but any edge with a maximum distance of 2.1 Å is considered covalent.

#### Nonbond energies:
These features measure nonbond energy potentials between nodes. 
For residue graphs, the pairwise sum of potentials for all atoms from each residue is used. Note that no distance cutoff is used and the radius of influence is assumed to be infinite, although the potentials tends to 0 at large distance. Also edges are only assigned within a given cutoff radius when graphs are created.
Nonbond energies are set to 0 for any atom pairs (on the same chain) that are within a cutoff radius of 3.6 Å, as these are assumed to be covalent neighbors or linked by no more than 2 covalent bonds (i.e. 1-3 pairs).

- `electrostatic`: Electrostatic potential (also known as Coulomb potential) between two nodes, calculated using interatomic distances and charges of each atom (float).
- `vanderwaals`: Van der Waals potential (also known as Lennard-Jones potential) between two nodes, calculated using interatomic distance/s and a list of atoms with vanderwaals parameters (`deeprankcore.domain.forcefield.protein-allhdg5-4_new`). Atom pairs within a cutoff radius of 4.2 Å (but above 3.6 Å) are assumed to be separated by separated by exactly 2 covalent bonds (i.e. 1-4 pairs) and use a set of lower energy parameters.
