from deeprank_gnn.models.amino_acid import AminoAcid
from deeprank_gnn.models.polarity import Polarity


alanine = AminoAcid("Alanine", "ALA", "A")
cysteine = AminoAcid("Cysteine", "CYS", "C")
aspartate = AminoAcid("Aspartate", "ASP", "D")
glutamate = AminoAcid("Glutamate", "GLU", "E")
phenylalanine = AminoAcid("Phenylalanine", "PHE", "F")
glycine = AminoAcid("Glycine", "GLY", "G")
histidine = AminoAcid("Histidine", "HIS", "H")
isoleucine = AminoAcid("Isoleucine", "ILE", "I")
leucine = AminoAcid("Leucine", "LEU", "L")
lysine = AminoAcid("Lysine", "LYS", "K")
methionine = AminoAcid("Methionine", "MET", "M")
proline = AminoAcid("Proline", "PRO", "P")
serine = AminoAcid("Serine", "SER", "S")
threonine = AminoAcid("Threonine", "THR", "T")
tryptophan = AminoAcid("Tryptophan", "TRP", "W")
tyrosine = AminoAcid("Tyrosine", "TYR", "Y")
valine = AminoAcid("Valine", "VAL", "V")
selenocysteine = AminoAcid("Selenocysteine", "SEC", "U")
pyrrolysine = AminoAcid("Pyrrolysine", "PYL", "O")
arginine = AminoAcid("Arginine", "ARG", "R")
asparagine = AminoAcid("Asparagine", "ASN", "N")
glutamine = AminoAcid("Glutamine", "GLN", "Q")


amino_acids = [alanine, arginine, asparagine, aspartate, cysteine, glutamate, glutamine, glycine,
               histidine, isoleucine, leucine, lysine, methionine, phenylalanine, proline, serine,
               threonine, tryptophan, tyrosine, valine]

amino_acid_charges = {

    cysteine: -0.64,
    histidine: -0.29,
    asparagine: -1.22,
    glutamine: -1.22,
    serine: -0.80,
    threonine: -0.80,
    tyrosine: -0.80,
    tryptophan: -0.79,
    alanine: -0.37,
    phenylalanine: -0.37,
    glycine: -0.37,
    isoleucine: -0.37,
    valine: -0.37,
    methionine: -0.37,
    proline: 0.0,
    leucine: -0.37,
    glutamate: -1.37,
    aspartate: -1.37,
    lysine: -0.36,
    arginine: -1.65
}


amino_acid_polarities = {

    cysteine: Polarity.POLAR,
    histidine: Polarity.POLAR,
    asparagine: Polarity.POLAR,
    glutamine: Polarity.POLAR,
    serine: Polarity.POLAR,
    threonine: Polarity.POLAR,
    tyrosine: Polarity.POLAR,
    tryptophan: Polarity.POLAR,
    alanine: Polarity.APOLAR,
    phenylalanine: Polarity.APOLAR,
    glycine: Polarity.APOLAR,
    isoleucine: Polarity.APOLAR,
    valine: Polarity.APOLAR,
    methionine: Polarity.APOLAR,
    proline: Polarity.APOLAR,
    leucine: Polarity.APOLAR,
    glutamate: Polarity.NEGATIVE_CHARGE,
    aspartate: Polarity.NEGATIVE_CHARGE,
    lysine: Polarity.POSITIVE_CHARGE,
    arginine: Polarity.POSITIVE_CHARGE
}
