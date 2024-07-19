from typing import Literal

from deeprank2.molstruct.aminoacid import AminoAcid, Polarity

# All info below sourced from above websites in December 2022 and summarized in deeprank2/domain/aminoacid_summary.xlsx

# Charge is calculated from summing all atoms in the residue (from ./deeprank2/domain/forcefield/protein-allhdg5-5_new.top).
# This results in the expected charge of 0 for all POLAR and NONPOLAR residues, +1 for POSITIVE residues and -1 for NEGATIVE residues.
# Note that SER, THR, and TYR lead to a charge of ~1e-16. A rounding error is assumed in these cases and they are set to 0.

# Sources for Polarity:
#   1) https://www.britannica.com/science/amino-acid/Standard-amino-acids
#   2) https://www.shimadzu.co.jp/aboutus/ms_r/archive/files/AminoAcidTable.pdf
#   3) https://en.wikipedia.org/wiki/Amino_acid
#   3) https://nld.promega.com/resources/tools/amino-acid-chart-amino-acid-structure/
#   5) https://ib.bioninja.com.au/standard-level/topic-2-molecular-biology/24-proteins/amino-acids.html
#   6) print book: "Biology", by Campbell & Reece, 6th ed, ISBN: 0-201-75054-6
# Sources 1, 2, and 6 sources agree on every amino acid and are used below.
# The other sources have some minor discrepancies compared to this and are commented inline.

# Source for size:
#   https://www.shimadzu.co.jp/aboutus/ms_r/archive/files/AminoAcidTable.pdf

# Sources for mass and pI:
#   1) https://www.sigmaaldrich.com/NL/en/technical-documents/technical-article/protein-biology/protein-structural-analysis/amino-acid-reference-chart
#   2) https://www.shimadzu.co.jp/aboutus/ms_r/archive/files/AminoAcidTable.pdf
#   3) https://www.nectagen.com/reference-data/ingredients/amino-acids
# Discrepancies of <0.1 (for either property) between sources are ignored.
# Two instances (LYS and THR) have a larger discrepancy for pI in 1/3 sources; majority rule is implemented (and outlier is indicated in inline comment).

# Sources for hydrogen bond donors and acceptors:
#   1) https://foldit.fandom.com/wiki/Sidechain_Bonding_Gallery
#   2) https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/charge/

# For selenocysteine and pyrrolysine only few sources detailing some of their properties could be found.
# Whatever sources were found (or lack thereof) are indicated in inline comments, but the reliability is much lower than for the canonical amino acids.
# Also, the rest of the package is not expecting these, so they removed from amino_acids at the bottom of this file.

alanine = AminoAcid(
    "Alanine",
    "ALA",
    "A",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=1,
    mass=71.1,
    pI=6.00,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=0,
)

cysteine = AminoAcid(
    "Cysteine",
    "CYS",
    "C",
    charge=0,
    polarity=Polarity.POLAR,  # source 3: "special case"; source 5: nonpolar
    # polarity of C is generally considered ambiguous: https://chemistry.stackexchange.com/questions/143142/why-is-the-amino-acid-cysteine-classified-as-polar
    size=2,
    mass=103.2,
    pI=5.07,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=1,
)

selenocysteine = AminoAcid(
    "Selenocysteine",
    "SEC",
    "U",
    charge=0,
    polarity=Polarity.POLAR,  # source 3: "special case"
    size=2,  # source: https://en.wikipedia.org/wiki/Selenocysteine
    mass=150.0,  # only from source 3
    pI=5.47,  # only from source 3
    hydrogen_bond_donors=1,  # unconfirmed
    hydrogen_bond_acceptors=2,  # unconfirmed
    index=cysteine.index,
)

aspartate = AminoAcid(
    "Aspartate",
    "ASP",
    "D",
    charge=-1,
    polarity=Polarity.NEGATIVE,
    size=4,
    mass=115.1,
    pI=2.77,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=4,
    index=2,
)

glutamate = AminoAcid(
    "Glutamate",
    "GLU",
    "E",
    charge=-1,
    polarity=Polarity.NEGATIVE,
    size=5,
    mass=129.1,
    pI=3.22,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=4,
    index=3,
)

phenylalanine = AminoAcid(
    "Phenylalanine",
    "PHE",
    "F",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=7,
    mass=147.2,
    pI=5.48,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=4,
)

glycine = AminoAcid(
    "Glycine",
    "GLY",
    "G",
    charge=0,
    polarity=Polarity.NONPOLAR,  # source 3: "special case"
    size=0,
    mass=57.1,
    pI=5.97,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=5,
)

histidine = AminoAcid(
    "Histidine",
    "HIS",
    "H",
    charge=1,
    polarity=Polarity.POSITIVE,
    size=6,
    mass=137.1,
    pI=7.59,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=1,
    # both position 7 and 10 can serve as either donor or acceptor (depending on tautomer), but any single His will have exactly one donor and one acceptor
    # (see https://foldit.fandom.com/wiki/Histidine)
    index=6,
)

isoleucine = AminoAcid(
    "Isoleucine",
    "ILE",
    "I",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=4,
    mass=113.2,
    pI=6.02,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=7,
)

lysine = AminoAcid(
    "Lysine",
    "LYS",
    "K",
    charge=1,
    polarity=Polarity.POSITIVE,
    size=5,
    mass=128.2,
    pI=9.74,  # 9.60 in source 3
    hydrogen_bond_donors=3,
    hydrogen_bond_acceptors=0,
    index=8,
)

pyrrolysine = AminoAcid(
    "Pyrrolysine",
    "PYL",
    "O",
    charge=0,  # unconfirmed
    polarity=Polarity.POLAR,  # based on having both H-bond donors and acceptors
    size=13,  # source: https://en.wikipedia.org/wiki/Pyrrolysine
    mass=255.32,  # from source 3
    pI=7.394,  # rough estimate from https://rstudio-pubs-static.s3.amazonaws.com/846259_7a9236df54e6410a972621590ecdcfcb.html
    hydrogen_bond_donors=1,  # unconfirmed
    hydrogen_bond_acceptors=4,  # unconfirmed
    index=lysine.index,
)

leucine = AminoAcid(
    "Leucine",
    "LEU",
    "L",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=4,
    mass=113.2,
    pI=5.98,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=9,
)

methionine = AminoAcid(
    "Methionine",
    "MET",
    "M",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=4,
    mass=131.2,
    pI=5.74,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=10,
)

asparagine = AminoAcid(
    "Asparagine",
    "ASN",
    "N",
    charge=0,
    polarity=Polarity.POLAR,
    size=4,
    mass=114.1,
    pI=5.41,
    hydrogen_bond_donors=2,
    hydrogen_bond_acceptors=2,
    index=11,
)

proline = AminoAcid(
    "Proline",
    "PRO",
    "P",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=3,
    mass=97.1,
    pI=6.30,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=12,
)

glutamine = AminoAcid(
    "Glutamine",
    "GLN",
    "Q",
    charge=0,
    polarity=Polarity.POLAR,
    size=5,
    mass=128.1,
    pI=5.65,
    hydrogen_bond_donors=2,
    hydrogen_bond_acceptors=2,
    index=13,
)

arginine = AminoAcid(
    "Arginine",
    "ARG",
    "R",
    charge=1,
    polarity=Polarity.POSITIVE,
    size=7,
    mass=156.2,
    pI=10.76,
    hydrogen_bond_donors=5,
    hydrogen_bond_acceptors=0,
    index=14,
)

serine = AminoAcid(
    "Serine",
    "SER",
    "S",
    charge=0,
    polarity=Polarity.POLAR,
    size=2,
    mass=87.1,
    pI=5.68,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=2,
    index=15,
)

threonine = AminoAcid(
    "Threonine",
    "THR",
    "T",
    charge=0,
    polarity=Polarity.POLAR,
    size=3,
    mass=101.1,
    pI=5.60,  # 6.16 in source 2
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=2,
    index=16,
)

valine = AminoAcid(
    "Valine",
    "VAL",
    "V",
    charge=0,
    polarity=Polarity.NONPOLAR,
    size=3,
    mass=99.1,
    pI=5.96,
    hydrogen_bond_donors=0,
    hydrogen_bond_acceptors=0,
    index=17,
)

tryptophan = AminoAcid(
    "Tryptophan",
    "TRP",
    "W",
    charge=0,
    polarity=Polarity.NONPOLAR,  # source 4: polar
    size=10,
    mass=186.2,
    pI=5.89,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=0,
    index=18,
)

tyrosine = AminoAcid(
    "Tyrosine",
    "TYR",
    "Y",
    charge=-0.0,
    polarity=Polarity.POLAR,  # source 3: nonpolar
    size=8,
    mass=163.2,
    pI=5.66,
    hydrogen_bond_donors=1,
    hydrogen_bond_acceptors=1,
    index=19,
)


# Including selenocysteine and pyrrolysine in the future will require some work to be done on the package.
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
    # selenocysteine,
    # pyrrolysine,
]

amino_acids_by_code = {amino_acid.three_letter_code: amino_acid for amino_acid in amino_acids}
amino_acids_by_letter = {amino_acid.one_letter_code: amino_acid for amino_acid in amino_acids}
amino_acids_by_name = {amino_acid.name: amino_acid for amino_acid in amino_acids}


def convert_aa_nomenclature(aa: str, output_format: Literal[0, 1, 3] = 0) -> str:
    """Converts amino acid nomenclatures.

    Conversions are possible between the standard 1-letter codes, 3-letter
    codes, and full names of amino acids.

    Args:
        aa: The amino acid to be converted in any of its formats. The length of the string is used to determine which format is used.
        output_format: Nomenclature style to return:
            0 (default) returns the full name,
            1 returns the 1-letter code,
            3 returns the 3-letter code.

    Raises:
        ValueError: If aa is not recognized or an invalid output format was given

    Returns:
        Amino acid identifier in the selected nomenclature system.
    """
    try:
        if len(aa) == 1:
            aa: AminoAcid = next(entry for entry in amino_acids if entry.one_letter_code.lower() == aa.lower())
        elif len(aa) == 3:  # noqa:PLR2004
            aa: AminoAcid = next(entry for entry in amino_acids if entry.three_letter_code.lower() == aa.lower())
        else:
            aa: AminoAcid = next(entry for entry in amino_acids if entry.name.lower() == aa.lower())
    except IndexError as e:
        msg = f"{aa} is not a valid amino acid."
        raise ValueError(msg) from e

    if not output_format:
        return aa.name
    if output_format == 3:  # noqa:PLR2004
        return aa.three_letter_code
    if output_format == 1:
        return aa.one_letter_code
    msg = f"output_format {output_format} not recognized. Must be set to 0 (amino acid name), 1 (one letter code), or 3 (three letter code)."
    raise ValueError(msg)
