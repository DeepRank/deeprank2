import numpy
from enum import Enum



class Polarity(Enum):
    "a value to express a residue's polarity"

    APOLAR = 0
    POLAR = 1
    NEGATIVE_CHARGE = 2
    POSITIVE_CHARGE = 3

    @property
    def onehot(self):
        t = numpy.zeros(4)
        t[self.value] = 1.0

        return t


class AminoAcid:
    "a value to represent one of the amino acids"

    def __init__( # pylint: disable=too-many-arguments
        self,
        name: str,
        three_letter_code: str,
        one_letter_code: str,
        charge: float,
        polarity: Polarity,
        size: int,
        count_hydrogen_bond_donors: int,
        count_hydrogen_bond_acceptors: int,
        index: int,
    ):
        """
        Args:
            name(str): unique name for the amino acid
            three_letter_code(str): code of the amino acid, as in PDB
            one_letter_code(str): letter of the amino acid, as in fasta
            charge(float, optional): the charge property of the amino acid
            polarity(deeprank polarity enum, optional): the polarity property of the amino acid
            size(int, optional): the number of heavy atoms in the side chain
            index(int, optional): the rank of the amino acid, used for computing one-hot encoding
        """

        self._name = name
        self._three_letter_code = three_letter_code
        self._one_letter_code = one_letter_code

        # these settings apply to the side chain
        self._size = size
        self._charge = charge
        self._polarity = polarity
        self._count_hydrogen_bond_donors = count_hydrogen_bond_donors
        self._count_hydrogen_bond_acceptors = count_hydrogen_bond_acceptors

        self._index = index

    @property
    def name(self) -> str:
        return self._name

    @property
    def three_letter_code(self) -> str:
        return self._three_letter_code

    @property
    def one_letter_code(self) -> str:
        return self._one_letter_code

    @property
    def onehot(self) -> numpy.ndarray:
        if self._index is None:
            raise ValueError(
                "amino acid {self._name} index is not set, thus no onehot can be computed"
            )

        # assumed that there are only 20 different amino acids
        a = numpy.zeros(20)
        a[self._index] = 1.0

        return a

    @property
    def count_hydrogen_bond_donors(self) -> int:
        return self._count_hydrogen_bond_donors

    @property
    def count_hydrogen_bond_acceptors(self) -> int:
        return self._count_hydrogen_bond_acceptors

    @property
    def charge(self) -> float:
        return self._charge

    @property
    def polarity(self) -> Polarity:
        return self._polarity

    @property
    def size(self) -> int:
        return self._size

    @property
    def index(self) -> int:
        return self._index

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, type(self)) and other.name == self.name

    def __repr__(self):
        return self._name


alanine = AminoAcid(
    "Alanine",
    "ALA",
    "A",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=1,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0,
    index=0)

cysteine = AminoAcid(
    "Cysteine",
    "CYS",
    "C",
    charge=-0.64,
    polarity=Polarity.POLAR,
    size=2, 
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0,
    index=1)

aspartate = AminoAcid(
    "Aspartate",
    "ASP",
    "D",
    charge=-1.37,
    polarity=Polarity.NEGATIVE_CHARGE,
    size=4,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=4,
    index=2)

glutamate = AminoAcid(
    "Glutamate",
    "GLU",
    "E",
    charge=-1.37,
    polarity=Polarity.NEGATIVE_CHARGE,
    size=5,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=4,
    index=3)

phenylalanine = AminoAcid(
    "Phenylalanine",
    "PHE",
    "F",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=7,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0,
    index=4)

glycine = AminoAcid(
    "Glycine",
    "GLY",
    "G",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=0,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0,
    index=5)

histidine = AminoAcid(
    "Histidine",
    "HIS", 
    "H",
    charge=-0.29,
    polarity=Polarity.POLAR,
    size=6,
    count_hydrogen_bond_donors=2,
    count_hydrogen_bond_acceptors=2,
    index=6)

isoleucine = AminoAcid(
    "Isoleucine",
    "ILE", 
    "I",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=4,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0,
    index=7)

leucine = AminoAcid(
    "Leucine",
    "LEU",
    "L",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=4,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0,
    index=8)

lysine = AminoAcid(
    "Lysine",
    "LYS",
    "K",
    charge=-0.36,
    polarity=Polarity.POSITIVE_CHARGE,
    size=5,
    count_hydrogen_bond_donors=3,
    count_hydrogen_bond_acceptors=0,
    index=9)

methionine = AminoAcid(
    "Methionine",
    "MET",
    "M",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=4,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0,
    index=10)

proline = AminoAcid(
    "Proline",
    "PRO",
    "P",
    charge=0.0,
    polarity=Polarity.APOLAR,
    size=3,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0,
    index=11)

serine = AminoAcid(
    "Serine",
    "SER",
    "S",
    charge=-0.80,
    polarity=Polarity.POLAR,
    size=2,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=2,
    index=12)

threonine = AminoAcid(
    "Threonine",
    "THR",
    "T",
    charge=-0.80,
    polarity=Polarity.POLAR,
    size=3,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=2,
    index=13)

tryptophan = AminoAcid(
    "Tryptophan",
    "TRP",
    "W",
    charge=-0.79,
    polarity=Polarity.POLAR,
    size=10,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=0,
    index=14)

tyrosine = AminoAcid(
    "Tyrosine",
    "TYR",
    "Y",
    charge=-0.80,
    polarity=Polarity.POLAR,
    size=8,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=1,
    index=15)

valine = AminoAcid(
    "Valine",
    "VAL",
    "V",
    charge=-0.37,
    polarity=Polarity.APOLAR,
    size=3,
    count_hydrogen_bond_donors=0,
    count_hydrogen_bond_acceptors=0,
    index=16)

selenocysteine = AminoAcid(
    "Selenocysteine",
    "SEC",
    "U",
    charge=0.0,
    polarity=Polarity.POLAR,
    size=2,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=2,
    index=cysteine.index)

pyrrolysine = AminoAcid(
    "Pyrrolysine",
    "PYL",
    "O",
    charge=0.0,
    polarity=Polarity.POLAR,
    size=13,
    count_hydrogen_bond_donors=1,
    count_hydrogen_bond_acceptors=4,
    index=lysine.index)

arginine = AminoAcid(
    "Arginine",
    "ARG",
    "R",
    charge=-1.65,
    polarity=Polarity.POSITIVE_CHARGE,
    size=7,
    count_hydrogen_bond_donors=5,
    count_hydrogen_bond_acceptors=0,
    index=17)
asparagine = AminoAcid(
    "Asparagine",
    "ASN",
    "N",
    charge=-1.22,
    polarity=Polarity.POLAR,
    size=4,
    count_hydrogen_bond_donors=2,
    count_hydrogen_bond_acceptors=2,
    index=18)

glutamine = AminoAcid(
    "Glutamine",
    "GLN",
    "Q",
    charge=-1.22,
    polarity=Polarity.POLAR,
    size=5,
    count_hydrogen_bond_donors=2,
    count_hydrogen_bond_acceptors=2,
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
    valine]
