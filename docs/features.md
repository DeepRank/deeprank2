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
These features only exist in atomic level graphs.

- `atom_type`: One hot encoding of the atomic element. Options are: C, O, N, S, P, H.
- `atom_charge`: Atomic charge in Coulomb (float). Taken from `deeprankcore.domain.forcefield.patch.top`.
- `pdb_occupancy`: Proportion of structures where the atom was detected at this position (float). In some cases a single atom was detected at different positions, in which case separate structures exist whose occupancies sum to 1. Only the highest occupancy atom is used by deeprankcore. 

#### Residue properties:
- `res_type`: One hot encoding of the amino acid residue (size 20).
- `polarity`: One hot encoding of the polarity of the amino acid (options: NONPOLAR, POLAR, NEGATIVE, POSITIVE). Note that sources vary on the polarity for few of the amino acids; see detailed information in `deeprankcore.domain.aminoacidlist.py`.
- `res_size`: The number of non-hydrogen atoms in the side chain (int). 
- `res_mass`: The (average) residue mass in Da (float).
- `res_charge`: The charge of the residue (in fully protonated state) in Coulomb (int). Charge is calculated from summing all atoms in the residue, which results in a charge of 0 for all polar and nonpolar residues, +1 for positive residues and -1 for negative residues.
- `res_pI`: The isolectric point, i.e. the pH at which the molecule has no net electric charge (float).

- `hb_donors` / `hb_acceptors`: The number of hydrogen bond donor/acceptor atoms in the residue (int). Hydrogen bonds are noncovalent intermolecular interactions formed between an hydrogen atom (partially positively charged) bound to a small, highly electronegative atom (O, N, F) with an unshared electron pair.

#### Properties related to variant residues:
These features are only used in SingleVariantQueries.

- `variant_res`: One hot encoding of variant amino acid (size 20).
- `diff_charge`, `diff_polarity`, `diff_size`, `diff_mass`, `diff_pI`, `diff_hb_donors`, `diff_hb_acceptors`: Subtraction of the wildtype value of indicated feature from the variant value. For example, if the variant has 4 hb_donors and the wildtype has 5, then `diff_hb_donors == -1`.

### Conservation features: `deeprankcore.features.conservation`

- `pssm`: Position-specific scoring matrix (also known as position weight matrix, PWM) values relative to the residue, is a score of the conservation of the amino acid along all 20 amino acids. 
- `info_content`: Information content: difference between the given PSSM for an amino acid and a uniform distribution (float).
- `conservation` (only used in SingleVariantQueries): Conservation of the wild type amino acid (float). *More details required.*
- `diff_conservation` (only used in SingleVariantQueries): Subtraction of wildtype conservation from the variant conservation (float). 

### `deeprankcore.features.exposure`

- `res_depth`: Average distance to surface for all atoms in a residue. It can only be calculated per residue, not per atom. So for atomic graphs, every atom gets its residue's value. Computed using `Bio.PDB.ResidueDepth`, in Ångström. Float value. 
- `hse`: Half Sphere exposure (HSE) measures how buried amino acid residues are in a protein. It is found by counting the number of amino acid neighbors within two half spheres of chosen radius around the amino acid. It can only be calculated per residue, not per atom. So for atomic graphs, every atom gets its residue's value. It is calculated using biopython, so for more details see [Bio.PDB.HSExposure](https://biopython.org/docs/dev/api/Bio.PDB.HSExposure.html#module-Bio.PDB.HSExposure) biopython module. Array of float values of length 3.
  
### `deeprankcore.features.surfacearea`

- `sasa`: Solvent-Accessible Surface Area. It is defined as the surface characterized around a protein by a hypothetical centre of a solvent sphere with the van der Waals contact surface of the molecule. Computed using FreeSASA (https://freesasa.github.io/doxygen/Geometry.html), in square Ångström. Float value. 
- `bsa`: the Buried interfacial Surface Area is the area of the protein that only gets exposed in monomeric state. It measures the size of the interface in a protein-protein. Computed using FreeSASA, in square Ångström. Float value. 

## Edge features

### `deeprankcore.features.contact`

- `same_res`: Only for atomic graph, 1 if the edge connects two atoms beloging to the same residue, otherwise 0.  
- `same_chain`: 1 if the edge connects two molecules beloging to the same chain, otherwise 0.  
- `distance`: Interatomic distance between atoms in Ångström. It is computed from the xyz atomic coordinates taken from the .pdb file. In the residue graph case, the minimum distance between the atoms of the first residue and the atoms from the second one is considered. Float value. 
- `covalent`: 1 if there is a covalent bond between the two molecules, otherwise 0. A bond is considered covalent if its length is less than 2.1 Ångström.
- `electrostatic`: Coulomb (electrostatic) potential, given the interatomic distance/s and charge/s of the atoms. There's no distance cutoff here. The radius of influence is assumed to infinite. Float value. 
- `vanderwaals`: Lennard-Jones potentials, given interatomic distance/s and a list of atoms with vanderwaals parameters. There's no distance cutoff here. The radius of influence is assumed to infinite. Float value.
