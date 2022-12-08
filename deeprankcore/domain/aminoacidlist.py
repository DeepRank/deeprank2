from deeprankcore.molstruct.aminoacid import AminoAcid,Polarity

# Mass and pI confirmed from 3 independent sources for canonical amino acids. Discrepancies of <0.1 for either are ignored.
# Two instances (K and T) have a larger discrepancy for pI in 1/3 sources; majority rule is implemented (and outlier is indicated in inline comment)
# Sources:
#   1) https://www.sigmaaldrich.com/NL/en/technical-documents/technical-article/protein-biology/protein-structural-analysis/amino-acid-reference-chart
#   2) https://www.shimadzu.co.jp/aboutus/ms_r/archive/files/AminoAcidTable.pdf
#   3) https://www.nectagen.com/reference-data/ingredients/amino-acids
# Sources for selenocysteine and pyrrolysine sources are indicated in inline comments.


alanine = AminoAcid(
    "Alanine",
    "ALA",
    "A",
    propertyX = -0.37,
    polarity = Polarity.APOLAR,
    size = 1,
    mass = 71.1,
    pI = 6.00,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 0,
    index = 0)

cysteine = AminoAcid(
    "Cysteine",
    "CYS",
    "C",
    propertyX = -0.64,
    polarity = Polarity.POLAR,
    size = 2,
    mass = 103.2,
    pI = 5.07,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 0,
    index = 1)

selenocysteine = AminoAcid(
    "Selenocysteine",
    "SEC",
    "U",
    propertyX = 0.0,
    polarity = Polarity.POLAR,
    size = 2,
    mass = 150.0, # from source 3
    pI = 5.47, # from source 3
    hydrogen_bond_donors = 1,
    hydrogen_bond_acceptors = 2,
    index = cysteine.index)

aspartate = AminoAcid(
    "Aspartate",
    "ASP",
    "D",
    propertyX = -1.37,
    polarity = Polarity.NEGATIVE_CHARGE,
    size = 4,
    mass = 115.1,
    pI = 2.77,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 4,
    index = 2)

glutamate = AminoAcid(
    "Glutamate",
    "GLU",
    "E",
    propertyX = -1.37,
    polarity = Polarity.NEGATIVE_CHARGE,
    size = 5,
    mass = 129.1,
    pI = 3.22,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 4,
    index = 3)

phenylalanine = AminoAcid(
    "Phenylalanine",
    "PHE",
    "F",
    propertyX = -0.37,
    polarity = Polarity.APOLAR,
    size = 7,
    mass = 147.2,
    pI = 5.48,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 0,
    index = 4)

glycine = AminoAcid(
    "Glycine",
    "GLY",
    "G",
    propertyX = -0.37,
    polarity = Polarity.APOLAR,
    size = 0,
    mass = 57.1,
    pI = 5.97,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 0,
    index = 5)

histidine = AminoAcid(
    "Histidine",
    "HIS",
    "H",
    propertyX = -0.29,
    polarity = Polarity.POLAR,
    size = 6,
    mass = 137.1,
    pI = 7.59,
    hydrogen_bond_donors = 2,
    hydrogen_bond_acceptors = 2,
    index = 6)

isoleucine = AminoAcid(
    "Isoleucine",
    "ILE",
    "I",
    propertyX = -0.37,
    polarity = Polarity.APOLAR,
    size = 4,
    mass = 113.2,
    pI = 6.02,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 0,
    index = 7)

lysine = AminoAcid(
    "Lysine",
    "LYS",
    "K",
    propertyX = -0.36,
    polarity = Polarity.POSITIVE_CHARGE,
    size = 5,
    mass = 128.2,
    pI = 9.74, # 9.60 in source 3
    hydrogen_bond_donors = 3,
    hydrogen_bond_acceptors = 0,
    index = 9)

pyrrolysine = AminoAcid(
    "Pyrrolysine",
    "PYL",
    "O",
    propertyX = 0.0,
    polarity = Polarity.POLAR,
    size = 13,
    mass = 255.32, # from source 3
    pI = 7.394, # rough estimate from https://rstudio-pubs-static.s3.amazonaws.com/846259_7a9236df54e6410a972621590ecdcfcb.html
    hydrogen_bond_donors = 1,
    hydrogen_bond_acceptors = 4,
    index = lysine.index)

leucine = AminoAcid(
    "Leucine",
    "LEU",
    "L",
    propertyX = -0.37,
    polarity = Polarity.APOLAR,
    size = 4,
    mass = 113.2,
    pI = 5.98,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 0,
    index = 8)

methionine = AminoAcid(
    "Methionine",
    "MET",
    "M",
    propertyX = -0.37,
    polarity = Polarity.APOLAR,
    size = 4,
    mass = 131.2,
    pI = 5.74,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 0,
    index = 10)

asparagine = AminoAcid(
    "Asparagine",
    "ASN",
    "N",
    propertyX = -1.22,
    polarity = Polarity.POLAR,
    size = 4,
    mass = 114.1,
    pI = 5.41,
    hydrogen_bond_donors = 2,
    hydrogen_bond_acceptors = 2,
    index = 18)

proline = AminoAcid(
    "Proline",
    "PRO",
    "P",
    propertyX = 0.0,
    polarity = Polarity.APOLAR,
    size = 3,
    mass = 97.1,
    pI = 6.30,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 0,
    index = 11)

glutamine = AminoAcid(
    "Glutamine",
    "GLN",
    "Q",
    propertyX = -1.22,
    polarity = Polarity.POLAR,
    size = 5,
    mass = 128.1,
    pI = 5.65,
    hydrogen_bond_donors = 2,
    hydrogen_bond_acceptors = 2,
    index = 19)

arginine = AminoAcid(
    "Arginine",
    "ARG",
    "R",
    propertyX = -1.65,
    polarity = Polarity.POSITIVE_CHARGE,
    size = 7,
    mass = 156.2,
    pI = 10.76,
    hydrogen_bond_donors = 5,
    hydrogen_bond_acceptors = 0,
    index = 17)


serine = AminoAcid(
    "Serine",
    "SER",
    "S",
    propertyX = -0.80,
    polarity = Polarity.POLAR,
    size = 2,
    mass = 87.1,
    pI = 5.68,
    hydrogen_bond_donors = 1,
    hydrogen_bond_acceptors = 2,
    index = 12)

threonine = AminoAcid(
    "Threonine",
    "THR",
    "T",
    propertyX = -0.80,
    polarity = Polarity.POLAR,
    size = 3,
    mass = 101.1,
    pI = 5.60, # 6.16 in source 2
    hydrogen_bond_donors = 1,
    hydrogen_bond_acceptors = 2,
    index = 13)

valine = AminoAcid(
    "Valine",
    "VAL",
    "V",
    propertyX = -0.37,
    polarity = Polarity.APOLAR,
    size = 3,
    mass = 99.1,
    pI = 5.96,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 0,
    index = 16)

tryptophan = AminoAcid(
    "Tryptophan",
    "TRP",
    "W",
    propertyX = -0.79,
    polarity = Polarity.POLAR,
    size = 10,
    mass = 186.2,
    pI = 5.89,
    hydrogen_bond_donors = 1,
    hydrogen_bond_acceptors = 0,
    index = 14)

tyrosine = AminoAcid(
    "Tyrosine",
    "TYR",
    "Y",
    propertyX = -0.80,
    polarity = Polarity.POLAR,
    size = 8,
    mass = 163.2,
    pI = 5.66,
    hydrogen_bond_donors = 1,
    hydrogen_bond_acceptors = 1,
    index = 15)


amino_acids = [
    alanine,
    arginine,
    asparagine,
    aspartate,
    cysteine,
    glutamate,
    glutamine,
    glycine,
    histidine,
    isoleucine,
    leucine,
    lysine,
    methionine,
    phenylalanine,
    proline,
    serine,
    threonine,
    tryptophan,
    tyrosine,
    valine,
    selenocysteine,
    pyrrolysine]
