import os
import numpy as np

from deeprank_gnn.models.structure import Residue, Chain
from deeprank_gnn.models.conservation import ConservationRow, ConservationTable
from deeprank_gnn.domain.amino_acid import amino_acids


amino_acids_by_letter = {amino_acid.one_letter_code: amino_acid for amino_acid in amino_acids}


def parse_pssm(file_, chain):
    """Read the PSSM data."""

    conservation_rows = {}

    header = next(file_).split()
    column_indices = {column_name.strip(): index for index, column_name in enumerate(header)}

    for line in file_:
        row = line.split()

        amino_acid = amino_acids_by_letter[row[column_indices["pdbresn"]]]

        pdb_residue_number_string = row[column_indices["pdbresi"]]
        if pdb_residue_number_string[-1].isalpha():

            pdb_residue_number = int(pdb_residue_number_string[:-1])
            pdb_insertion_code = pdb_residue_number_string[-1]
        else:
            pdb_residue_number = int(pdb_residue_number_string)
            pdb_insertion_code = None

        residue = Residue(chain, pdb_residue_number, amino_acid, pdb_insertion_code)

        information_content = float(row[column_indices["IC"]])
        conservations = {amino_acid: float(row[column_indices[amino_acid.one_letter_code]]) for amino_acid in amino_acids}

        conservation_rows[residue] = ConservationRow(conservations, information_content)

    return ConservationTable(conservation_rows)


def add_pssms(structure, pssm_paths):
    for chain in structure.chains:
        path = pssm_paths[chain.id]
        with open(path, 'rt') as f:
            chain.pssm = parse_pssm(f, chain)

