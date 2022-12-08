from deeprankcore.molstruct.aminoacid import AminoAcid, Polarity


alanine = AminoAcid(
    "Alanine",
    "ALA",
    "A",
    propertyX=-0.37,
    polarity=Polarity.APOLAR,
    size=1,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=0)

cysteine = AminoAcid(
    "Cysteine",
    "CYS",
    "C",
    propertyX=-0.64,
    polarity=Polarity.POLAR,
    size=2, 
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=1)

selenocysteine = AminoAcid(
    "Selenocysteine",
    "SEC",
    "U",
    propertyX=0.0,
    polarity=Polarity.POLAR,
    size=2,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=2,
    index=cysteine.index)

aspartate = AminoAcid(
    "Aspartate",
    "ASP",
    "D",
    propertyX=-1.37,
    polarity=Polarity.NEGATIVE_CHARGE,
    size=4,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=4,
    index=2)

glutamate = AminoAcid(
    "Glutamate",
    "GLU",
    "E",
    propertyX=-1.37,
    polarity=Polarity.NEGATIVE_CHARGE,
    size=5,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=4,
    index=3)

phenylalanine = AminoAcid(
    "Phenylalanine",
    "PHE",
    "F",
    propertyX=-0.37,
    polarity=Polarity.APOLAR,
    size=7,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=4)

glycine = AminoAcid(
    "Glycine",
    "GLY",
    "G",
    propertyX=-0.37,
    polarity=Polarity.APOLAR,
    size=0,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=5)

histidine = AminoAcid(
    "Histidine",
    "HIS", 
    "H",
    propertyX=-0.29,
    polarity=Polarity.POLAR,
    size=6,
    hydrogen_bond_donors=2,
    hydrogen_bond_acceptors=2,
    index=6)

isoleucine = AminoAcid(
    "Isoleucine",
    "ILE", 
    "I",
    propertyX=-0.37,
    polarity=Polarity.APOLAR,
    size=4,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=7)

leucine = AminoAcid(
    "Leucine",
    "LEU",
    "L",
    propertyX=-0.37,
    polarity=Polarity.APOLAR,
    size=4,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=8)

lysine = AminoAcid(
    "Lysine",
    "LYS",
    "K",
    propertyX=-0.36,
    polarity=Polarity.POSITIVE_CHARGE,
    size=5,
    hydrogen_bond_donors=3,
    hydrogen_bond_acceptors=0,
    index=9)

pyrrolysine = AminoAcid(
    "Pyrrolysine",
    "PYL",
    "O",
    propertyX=0.0,
    polarity=Polarity.POLAR,
    size=13,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=4,
    index=lysine.index)

methionine = AminoAcid(
    "Methionine",
    "MET",
    "M",
    propertyX=-0.37,
    polarity=Polarity.APOLAR,
    size=4,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=10)

proline = AminoAcid(
    "Proline",
    "PRO",
    "P",
    propertyX=0.0,
    polarity=Polarity.APOLAR,
    size=3,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=11)

serine = AminoAcid(
    "Serine",
    "SER",
    "S",
    propertyX=-0.80,
    polarity=Polarity.POLAR,
    size=2,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=2,
    index=12)

threonine = AminoAcid(
    "Threonine",
    "THR",
    "T",
    propertyX=-0.80,
    polarity=Polarity.POLAR,
    size=3,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=2,
    index=13)

tryptophan = AminoAcid(
    "Tryptophan",
    "TRP",
    "W",
    propertyX=-0.79,
    polarity=Polarity.POLAR,
    size=10,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=0,
    index=14)

tyrosine = AminoAcid(
    "Tyrosine",
    "TYR",
    "Y",
    propertyX=-0.80,
    polarity=Polarity.POLAR,
    size=8,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=1,
    index=15)

valine = AminoAcid(
    "Valine",
    "VAL",
    "V",
    propertyX=-0.37,
    polarity=Polarity.APOLAR,
    size=3,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=16)

arginine = AminoAcid(
    "Arginine",
    "ARG",
    "R",
    propertyX=-1.65,
    polarity=Polarity.POSITIVE_CHARGE,
    size=7,
    hydrogen_bond_donors=5,
    hydrogen_bond_acceptors=0,
    index=17)

asparagine = AminoAcid(
    "Asparagine",
    "ASN",
    "N",
    propertyX=-1.22,
    polarity=Polarity.POLAR,
    size=4,
    hydrogen_bond_donors=2,
    hydrogen_bond_acceptors=2,
    index=18)

glutamine = AminoAcid(
    "Glutamine",
    "GLN",
    "Q",
    propertyX=-1.22,
    polarity=Polarity.POLAR,
    size=5,
    hydrogen_bond_donors=2,
    hydrogen_bond_acceptors=2,
    index=19)

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
