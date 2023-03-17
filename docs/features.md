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


## Node features modules

### `deeprankcore.features.components`
These features relate to the chemical components (atoms and amino acid residues) of which the graph is composed. Detailed information and descrepancies between sources are described can be found in deeprankcore.domain.aminoacidlist.py.
For atomic graphs, when features relate to residue then _all_ atoms of one residue receive the feature value for that residue.

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
These features relate to the conservation state of individual residues.
For atomic graphs: _all_ atoms of one residue receive the feature value for that residue.

| feature | description | type | notes | restrictions | 
| --- | --------- | --- | --- | --- | --- |
| `pssm` | Position weight matrix (PWM) values relative to the residue | Array of floats of length 20 | The conservation of the residue along all the 20 amino acids |
| `info_content` | Difference between PWM and unoform distribution | float |
| `conservation` | Conservation of the wild type residue | float | | SingleResidueVariant graphs only |
| `diff_conservation` | Difference between conservation of the wild type and variant residue | float | | SingleResidueVariant graphs only |


### `deeprankcore.features.exposure`
These features relate to the exposure of residues to the surface, and are computed using [biopython](https://biopython.org/docs/1.75/api/Bio.PDB.html). Note that these features can only be calculated per residue and not per atom.
For atomic graphs: _all_ atoms of one residue receive the feature value for that residue.
  
| feature | description | type | notes | 
| --- | --------- | --- | --- | --- | 
| `res_depth` | Distance to the surface in Å | float | Average distance for all atoms in residue. See also [`Bio.PDB.ResidueDepth`](https://biopython.org/docs/1.75/api/Bio.PDB.ResidueDepth.html) | 
| `hse` | Half Sphere Exposure (HSE) | Array of floats of length 3 | Measures the buried-ness of a residues in a protein. It is found by counting the number of amino acid neighbors within two half spheres of chosen radius around the amino acid. See also [Bio.PDB.HSExposure](https://biopython.org/docs/dev/api/Bio.PDB.HSExposure.html) |


### `deeprankcore.features.surfacearea`
These features relate to the surface area of the residue, and are computed using [freesasa](https://freesasa.github.io) 
For atomic graphs: _all_ atoms of one residue receive the feature value for that residue.

| feature | description | type | notes | 
| --- | --------- | --- | --- | --- | 
| `sasa` | Solvent-Accessible Surface Area in Å^2 | float | Surface area characterized around a protein by a hypothetical centre of a solvent sphere with the van der Waals contact surface of the molecule. See also: https://freesasa.github.io/doxygen/Geometry.html.
| `bsa` | Buried interfacial Surface Area in Å^2 | float | Area of the protein that only gets exposed in monomeric state. It measures the size of the interface in a protein-protein. |


### `deeprankcore.features.irc`
These features relate to the inter-residue contacs (IRCs), i.e. the number of residues on the opposite chain within a cutoff distance of 5.5 Å. IRCs are found using the `get_contact_residues` function of [pdb2sql.interface](https://github.com/DeepRank/pdb2sql/blob/master/pdb2sql/interface.py)
For atomic graphs: _all_ atoms of one residue receive the feature value for that residue.

| feature | description | type | notes | restrictions
| --- | --------- | --- | --- | --- | 
| `irc_total` | Total inter-residue contacts | int | Number of residues on opposite chain within 5.5 Å distance | ProteinProteinInteraction graphs only |


## Edge features module

### `deeprankcore.features.contact`
These features relate to relationships between individual nodes.
For atomic graphs, when features relate to residue then _all_ atoms of one residue receive the feature value for that residue.

| feature | description | type | notes | restrictions | 
| --- | --------- | --- | --- | --- | --- |
| `same_res` | Whether both nodes are part of the same residue | bool | | atomic graphs only |
| `same_chain` | Whether both nodes are part of the same chain | bool |
| `distance` | Distance in Å | float | Computed using atomic coordinates from the .pdb file. For residue graphs, the minimum distance between atoms of each residue is used | 
| `covalent` | Whether the edge respresents a covalent bond | bool | Edges with a distance of <2.1 Å are considered covalent.
| `electrostatic` | Electrostatic potential energy (Coulomb potential) | float | Calculated from the interatomic distances and charges of the atoms. Note that no distance cutoff is implemented. | 
| `vanderwaals` | Van der Waals potential energy (Lennard-Jones potentials) | float | Calculated from the interatomic distances and the forcefield in deeprankcore/domain/forcefield/protein-allhdg5-4_new.param. Note that no distance cutoff is implemented. | 
