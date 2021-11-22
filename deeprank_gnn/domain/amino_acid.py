from deeprank_gnn.models.amino_acid import AminoAcid
from deeprank_gnn.models.polarity import Polarity


alanine = AminoAcid("Alanine", "ALA", "A", charge=-0.37, polarity=Polarity.APOLAR, index=0)
cysteine = AminoAcid("Cysteine", "CYS", "C", charge=-0.64, polarity=Polarity.POLAR, index=1)
aspartate = AminoAcid("Aspartate", "ASP", "D", charge=-1.37, polarity=Polarity.NEGATIVE_CHARGE, index=2)
glutamate = AminoAcid("Glutamate", "GLU", "E", charge=-1.37, polarity=Polarity.NEGATIVE_CHARGE, index=3)
phenylalanine = AminoAcid("Phenylalanine", "PHE", "F", charge=-0.37, polarity=Polarity.APOLAR, index=4)
glycine = AminoAcid("Glycine", "GLY", "G", charge=-0.37, polarity=Polarity.APOLAR, index=5)
histidine = AminoAcid("Histidine", "HIS", "H", charge=-0.29, polarity=Polarity.POLAR, index=6)
isoleucine = AminoAcid("Isoleucine", "ILE", "I", charge=-0.37, polarity=Polarity.APOLAR, index=7)
leucine = AminoAcid("Leucine", "LEU", "L", charge=-0.37, polarity=Polarity.APOLAR, index=8)
lysine = AminoAcid("Lysine", "LYS", "K", charge=-0.36, polarity=Polarity.POSITIVE_CHARGE, index=9)
methionine = AminoAcid("Methionine", "MET", "M", charge=-0.37, polarity=Polarity.APOLAR, index=10)
proline = AminoAcid("Proline", "PRO", "P", charge=0.0, polarity=Polarity.APOLAR, index=11)
serine = AminoAcid("Serine", "SER", "S", charge=-0.80, polarity=Polarity.POLAR, index=12)
threonine = AminoAcid("Threonine", "THR", "T", charge=-0.80, polarity=Polarity.POLAR, index=13)
tryptophan = AminoAcid("Tryptophan", "TRP", "W", charge=-0.79, polarity=Polarity.POLAR, index=14)
tyrosine = AminoAcid("Tyrosine", "TYR", "Y", charge=-0.80, polarity=Polarity.POLAR, index=15)
valine = AminoAcid("Valine", "VAL", "V", charge=-0.37, polarity=Polarity.APOLAR, index=16)
selenocysteine = AminoAcid("Selenocysteine", "SEC", "U")
pyrrolysine = AminoAcid("Pyrrolysine", "PYL", "O")
arginine = AminoAcid("Arginine", "ARG", "R", charge=-1.65, polarity=Polarity.POSITIVE_CHARGE, index=17)
asparagine = AminoAcid("Asparagine", "ASN", "N", charge=-1.22, polarity=Polarity.POLAR, index=18)
glutamine = AminoAcid("Glutamine", "GLN", "Q", charge=-1.22, polarity=Polarity.POLAR, index=19)


amino_acids = [alanine, arginine, asparagine, aspartate, cysteine, glutamate, glutamine, glycine,
               histidine, isoleucine, leucine, lysine, methionine, phenylalanine, proline, serine,
               threonine, tryptophan, tyrosine, valine]
