# Features

Features implemented in the code-base are defined in `deeprankcore.feature` subpackage.


## Custom Features

Users can add custom features by creating a new module and placing it in `deeprankcore.feature` subpackage. One requirement for any feature module is to implement an `add_features` function, as shown below. This will be used in `deeprankcore.models.query` to add the features to the nodes or edges of the graph.

```python
def add_features(
    pdb_path: str, graph: Graph,
    single_amino_acid_variant: Optional[SingleResidueVariant] = None
    ):
    pass
```


## Node features 

### `deeprankcore.features.components`

- Detailed information about sources used for the hard coded features (e.g., `polarity`) can be found in deeprankcore.domain.aminoacidlist.py.
- `atom_type`: Only for atomic graph, one hot encodes the type of atom (C, O, N, S, P, H).
- `pdb_occupancy`: Only for atomic graph, it represents the proportion of structures where the atom is detected at a given position. Sometimes a single atom can be detected at multiple positions, and in that case separate structures exist whose occupancies sum gives 1. Note that only the highest occupancy atom is used by deeprankcore. Float value.
- `atom_charge`: Only for atomic graph, atomic charge in Coulomb. Taken from deeprankcore.domain.forcefield.patch.top file. Float value.
- `res_type`: One hot encodes the type of amino acid (20).
- `charge`: The charge property of the amino acid. Charge is calculated from summing all atoms in the residue, which results in a charge of 0 for all polar and nonpolar residues, +1 for positive residues and -1 for negative residues.
- `polarity`: One hot encodes the polarity of the amino acid (nonpolar, polar, negative charge, positive charge).
- `res_size`: The number of heavy atoms in the side chain. Int value.
- `res_mass`: The average residue mass (i.e. mass of amino acid - H20) in Daltons. Float value.
- `res_pI`: The isolectric point, which represents the pH at which the molecule has no net electric charge. Float value.
- `hb_donors`, `hb_acceptors`: Represents the number of donor/acceptor atoms, from 0 to 5. Amino acids can have hydrogen donor/acceptor atoms in their side chain. Hydrogen Bonds (HB) are noncovalent intermolecular interactions formed between an hydrogen atom (partially positively charged) bound to a small, highly electronegative atom (O, N, F) with an unshared electron pair. In hydrogen bonds there is a distinction between the electronegative atoms (O, N, F) based on which one the hydrogen is covalently bonded to. Based on this, hydrogens can be named either acceptors or donators. Int value.
- `variant_res`: If a variant is present, one hot encodes the type of amino acid variant (20).
- `diff_charge`, `diff_polarity`, `diff_size`, `diff_mass`, `diff_pI`, `diff_hb_donors`, `diff_hb_acceptors`: If a variant is present, they represent the differences between the variant and the wild type amino acid in charge, polarity, size, mass, isoelectric point, donor/acceptor atoms.
  
### `deeprankcore.features.conservation`

- `pssm`: It represents the row of the position weight matrix (PWM) relative to the amino acid, which in turns represents the conservation of the amino acid along all the 20 amino acids. In the atomic graph case, it represents the PWM row relative to the amino acid to which the atom belongs. Array of int values of length 20.
- `info_content`: It represents how different a given PWM (row) for a given amino acid is from a uniform distribution. Float value.
- `conservation`: If a variant is present, it represents the conservation of the amino acid variant. Float value. 
- `diff_conservation`: If a variant is present, it represents the difference between the conservation of the variant and the conservation of the wild type amino acid. 

### `deeprankcore.features.exposure`

- `res_depth`: Average distance to surface for all atoms in a residue. It can only be calculated per residue, not per atom. So for atomic graphs, every atom gets its residue's value. Computed using `Bio.PDB.ResidueDepth`, in Angstrom. Float value. 
- `hse`: Half Sphere exposure (HSE) measures how buried amino acid residues are in a protein. It is found by counting the number of amino acid neighbors within two half spheres of chosen radius around the amino acid. It can only be calculated per residue, not per atom. So for atomic graphs, every atom gets its residue's value. It is calculated using biopython, so for more details see [Bio.PDB.HSExposure](https://biopython.org/docs/dev/api/Bio.PDB.HSExposure.html#module-Bio.PDB.HSExposure) biopython module. Array of float values of length 3.
  
### `deeprankcore.features.surfacearea`

- `sasa`: Solvent-Accessible Surface Area. It is defined as the surface characterized around a protein by a hypothetical centre of a solvent sphere with the van der Waals contact surface of the molecule. Computed using FreeSASA (https://freesasa.github.io/doxygen/Geometry.html), in square Angstrom. Float value. 
- `bsa`: the Buried interfacial Surface Area is the area of the protein that only gets exposed in monomeric state. It measures the size of the interface in a protein-protein. Computed using FreeSASA, in square Angstrom. Float value. 

## Edge features

### `deeprankcore.features.contact`

- `same_res`: Only for atomic graph, 1 if the edge connects two atoms beloging to the same residue, otherwise 0.  
- `same_chain`: 1 if the edge connects two molecules beloging to the same chain, otherwise 0.  
- `distance`: Interatomic distance between atoms in Angstrom. It is computed from the xyz atomic coordinates taken from the .pdb file. In the residue graph case, the minimum distance between the atoms of the first residue and the atoms from the second one is considered. Float value. 
- `covalent`: 1 if there is a covalent bond between the two molecules, otherwise 0. A bond is considered covalent if its length is less than 2.1 Angstrom.
- `electrostatic`: Coulomb (electrostatic) potential, given the interatomic distance/s and charge/s of the atoms. There's no distance cutoff here. The radius of influence is assumed to infinite. Float value. 
- `vanderwaals`: Lennard-Jones potentials, given interatomic distance/s and a list of atoms with vanderwaals parameters. There's no distance cutoff here. The radius of influence is assumed to infinite. Float value.
