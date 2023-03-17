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


## Predefined node features
Organized by module

### `deeprankcore.features.components`
These features relate to the chemical components (atoms and amino acid residues) of which the graph is composed. Detailed information and descrepancies between sources are described can be found in deeprankcore.domain.aminoacidlist.py.
For features related to residues in atomic graphs: _all_ atoms of one residue receive the feature value for that residue.

| feature | description | type | notes | restrictions | sources |
| --- | --------- | --- | --- | --- | --- |
| `atom_type` | Atomic element | one hot encoded: [C, O, N, S, P, H] | | atomic graphs only |
| `atom_charge` | Charge of the atom in Coulomb | float | Values can be found in deeprankcore.domain.forcefield.patch.top | atomic graphs only |
| `pdb_occupancy` | Proportion of structures where the atom was detected at this position | float | In some cases a single atom was detected at different positions, in which case separate structures exist whose occupancies sum to 1. Only the highest occupancy atom is used by deeprankcore. | atomic graphs only | 
| `res_type` | Amino acid residue | one hot encoded (size 20) | 
| `polarity` | Polarity of the amino acid | one hot encoded: [NONPOLAR, POLAR, NEGATIVE, POSITIVE] | Sources vary on the polarity for few of the amino acids; see detailed information in deeprankcore.domain.aminoacidlist.py | | 1-6 |
| `res_size` | Number of non-hydrogen atoms in the side chain | int | | | 2 |
| `res_mass` | Residue mass (average) in Da | float | Amino acid mass minus mass of H~2~0 | | 2, 7, 8 |
| `charge` | Charge of the residue (in fully protonated state) in Coulomb | float | Calculated by summing all atomic charges in the residue, resulting in a charge of 0 for all polar and nonpolar residues, +1 for positive residues and -1 for negative residues. |
| `res_pI` | Isolectric point of the residue (pH at which the molecule has no net electric charge) | float | Minor discrepancies between sources exists for few amino acids; see detailed information in deeprankcore.domain.aminoacidlist.py. | | 2, 7, 8 |
| `hb_donors` / `hb_acceptors` | Number of hydrogen bond donor/acceptor atoms in the residue | int | Hydrogen bonds (hb) are noncovalent intermolecular interactions formed between an hydrogen atom (partially positively charged) bound to a small, highly electronegative atom (O, N, F) with an unshared electron pair. | | 9, 10 |
| `variant_res` | Variant amino acid residue. | one hot encoded (size 20) | | SingleResidueVariant graphs only |
| `diff_charge` / `diff_polarity` / `diff_size` / `diff_mass` / `diff_pI` / `diff_hb_donors` / `diff_hb_acceptors` | Difference between the variant and the wild type equivalent for indicated feature, as described above | see above | see above | SingleResidueVariant graphs only|

Sources:
1. https://www.britannica.com/science/amino-acid/Standard-amino-acids
2. https://www.shimadzu.co.jp/aboutus/ms_r/archive/files/AminoAcidTable.pdf
3. https://en.wikipedia.org/wiki/Amino_acid
4. https://nld.promega.com/resources/tools/amino-acid-chart-amino-acid-structure/
5. https://ib.bioninja.com.au/standard-level/topic-2-molecular-biology/24-proteins/amino-acids.html
6. print book: "Biology", by Campbell & Reece, 6th ed, ISBN: 0-201-75054-6
7. https://www.sigmaaldrich.com/NL/en/technical-documents/technical-article/protein-biology/protein-structural-analysis/amino-acid-reference-chart
8. https://www.nectagen.com/reference-data/ingredients/amino-acids
9. https://foldit.fandom.com/wiki/Sidechain_Bonding_Gallery
10. https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/charge/


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
