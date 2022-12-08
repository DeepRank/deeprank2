from deeprankcore.molstruct.aminoacid import AminoAcid,Polarity

# Sources for Polarity (# few differences between sources are commented inline):
#   1) https://www.britannica.com/science/amino-acid/Standard-amino-acids
#   2) https://www.shimadzu.co.jp/aboutus/ms_r/archive/files/AminoAcidTable.pdf
#   3) https://en.wikipedia.org/wiki/Amino_acid
#   3) https://nld.promega.com/resources/tools/amino-acid-chart-amino-acid-structure/
#   5) https://ib.bioninja.com.au/standard-level/topic-2-molecular-biology/24-proteins/amino-acids.html
#   6) print book: "Biology", by Campbell & Reece, 6th ed, ISBN: 0-201-75054-6 

# Sources for mass and pI:
#   1) https://www.sigmaaldrich.com/NL/en/technical-documents/technical-article/protein-biology/protein-structural-analysis/amino-acid-reference-chart
#   2) https://www.shimadzu.co.jp/aboutus/ms_r/archive/files/AminoAcidTable.pdf
#   3) https://www.nectagen.com/reference-data/ingredients/amino-acids
# Sources for selenocysteine and pyrrolysine sources are indicated in inline comments.
# Discrepancies of <0.1 for either property are ignored.
# Two instances (K and T) have a larger discrepancy for pI in 1/3 sources; majority rule is implemented (and outlier is indicated in inline comment)

# Sources for hydrogen bond donors and acceptors:
#   1) https://foldit.fandom.com/wiki/Sidechain_Bonding_Gallery
#   2) https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/charge/


alanine = AminoAcid(
    "Alanine",
    "ALA",
    "A",
    propertyX = -0.37,
    polarity = Polarity.NONPOLAR,
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
    polarity = Polarity.POLAR, # source 3: "special case"; source 5: nonpolar
    # polarity of C is generally considered ambiguous: https://chemistry.stackexchange.com/questions/143142/why-is-the-amino-acid-cysteine-classified-as-polar
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
    polarity = Polarity.POLAR, # source 3: "special case"
    size = 2,
    mass = 150.0, # only from source 3
    pI = 5.47, # only from source 3
    hydrogen_bond_donors = 1, # unconfirmed
    hydrogen_bond_acceptors = 2, # unconfirmed
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
    polarity = Polarity.NONPOLAR,
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
    polarity = Polarity.NONPOLAR, # source 3: "special case"
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
    polarity = Polarity.POSITIVE_CHARGE,
    size = 6,
    mass = 137.1,
    pI = 7.59,
    hydrogen_bond_donors = 1,
    hydrogen_bond_acceptors = 1,
    # both position 7 and 10 can serve as either donor or acceptor (depending on tautomer), but any single His will have exactly one donor and one acceptor
    # (see https://foldit.fandom.com/wiki/Histidine)
    index = 6)

isoleucine = AminoAcid(
    "Isoleucine",
    "ILE",
    "I",
    propertyX = -0.37,
    polarity = Polarity.NONPOLAR,
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
    index = 8)

pyrrolysine = AminoAcid(
    "Pyrrolysine",
    "PYL",
    "O",
    propertyX = 0.0,
    polarity = Polarity.POLAR, # based on having both H-bond donors and acceptors 
    size = 13,
    mass = 255.32, # from source 3
    pI = 7.394, # rough estimate from https://rstudio-pubs-static.s3.amazonaws.com/846259_7a9236df54e6410a972621590ecdcfcb.html
    hydrogen_bond_donors = 1, # unconfirmed
    hydrogen_bond_acceptors = 4, # unconfirmed
    index = lysine.index)

leucine = AminoAcid(
    "Leucine",
    "LEU",
    "L",
    propertyX = -0.37,
    polarity = Polarity.NONPOLAR,
    size = 4,
    mass = 113.2,
    pI = 5.98,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 0,
    index = 9)

methionine = AminoAcid(
    "Methionine",
    "MET",
    "M",
    propertyX = -0.37,
    polarity = Polarity.NONPOLAR,
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
    index = 11)

proline = AminoAcid(
    "Proline",
    "PRO",
    "P",
    propertyX = 0.0,
    polarity = Polarity.NONPOLAR,
    size = 3,
    mass = 97.1,
    pI = 6.30,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 0,
    index = 12)

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
    index = 13)

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
    index = 14)

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
    index = 15)

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
    index = 16)

valine = AminoAcid(
    "Valine",
    "VAL",
    "V",
    propertyX = -0.37,
    polarity = Polarity.NONPOLAR,
    size = 3,
    mass = 99.1,
    pI = 5.96,
    hydrogen_bond_donors = 0,
    hydrogen_bond_acceptors = 0,
    index = 17)

tryptophan = AminoAcid(
    "Tryptophan",
    "TRP",
    "W",
    propertyX = -0.79,
    polarity = Polarity.NONPOLAR, # source 4: polar
    size = 10,
    mass = 186.2,
    pI = 5.89,
    hydrogen_bond_donors = 1,
    hydrogen_bond_acceptors = 0,
    index = 18)

tyrosine = AminoAcid(
    "Tyrosine",
    "TYR",
    "Y",
    propertyX = -0.80,
    polarity = Polarity.POLAR, # source 3: nonpolar
    size = 8,
    mass = 163.2,
    pI = 5.66,
    hydrogen_bond_donors = 1,
    hydrogen_bond_acceptors = 1,
    index = 19)


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
