import os
from enum import Enum
import logging

import pdb2sql
import numpy

from deeprank_gnn.tools.pssm import parse_pssm
from deeprank_gnn.tools import BioWrappers, BSA
from deeprank_gnn.tools.pdb import get_residue_contact_pairs, get_residue_distance
from deeprank_gnn.models.graph import Graph
from deeprank_gnn.domain.graph import EDGETYPE_INTERNAL, EDGETYPE_INTERFACE
from deeprank_gnn.domain.feature import *
from deeprank_gnn.domain.amino_acid import *


_log = logging.getLogger(__name__)


class Query:
    """ Represents one entity of interest, like a single residue variant or a protein-protein interface.

        Query objects are used to generate graphs from structures.
        objects of this class should be created before any model is loaded
    """

    def __init__(self, model_id, targets=None):
        """
            Args:
                model_id(str): the id of the model to load, usually a pdb accession code
                targets(dict, optional): target values associated with this query
        """

        self._model_id = model_id

        if targets is None:
            self._targets = {}
        else:
            self._targets = targets

    @property
    def model_id(self):
        return self._model_id

    @property
    def targets(self):
        return self._targets


class SingleResidueVariantAtomicQuery(Query):
    "creates an atomic graph for a single residue variant in a pdb file"

    def __init__(self, model_id, chain_id, residue_number, wildtype_amino_acid, variant_amino_acid, nonbonded_distance_cutoff=10.0, targets=None, insertion_code=None):
        Query.__init__(QueryType.SINGLE_RESIDUE_VARIANT, model_id, distance_cutoff, targets)

        self._chain_id = chain_id
        self._residue_number = residue_number
        self._insertion_code = insertion_code
        self._wildtype_amino_acid = wildtype_amino_acid
        self._variant_amino_acid = variant_amino_acid

    def __eq__(self, other):
        return type(self) == type(other) and self._residue == other._residue and \
            self._wildtype_amino_acid == other.wildtype_amino_acid and \
            self._variant_amino_acid == other.variant_amino_acid

    def __hash__(self):
        return hash((self._residue, self._wildtype_amino_acid, self._variant_amino_acid))

    def __repr__(self):
        residue_id = str(self._residue_number)
        if self._insertion_code is not None:
            residue_id += self._insertion_code

        return "SingleResidueVariantQuery({}{}{}:{}->{})".format(self._model_id, self._chain_id, residue_id,
                                                                 self._wildtype_amino_acid, self._variant_amino_acid)


class ProteinProteinInterfaceResidueQuery(Query):
    "a query that builds residue-based graphs, using the residues at a protein-protein interface"

    def __init__(self, pdb_path, chain_id1, chain_id2, pssm_paths=None,
                 interface_distance_cutoff=8.5, internal_distance_cutoff=3.0,
                 use_biopython=False, targets=None):
        """
            Args:
                pdb_path(str): the path to the pdb file
                chain_id1(str): the pdb chain identifier of the first protein of interest
                chain_id2(str): the pdb chain identifier of the second protein of interest
                pssm_paths(dict(str,str)): the paths to the pssm files, per chain identifier
                interface_distance_cutoff(float): max distance between two interacting residues of the two proteins
                internal_distance_cutoff(float): max distance between two interacting residues within the same protein
                use_biopython(bool): whether or not to use biopython tools
                targets(dict, optional): target values associated with this query
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._interface_distance_cutoff = interface_distance_cutoff
        self._internal_distance_cutoff = internal_distance_cutoff

        self._use_biopython = use_biopython

    def get_query_id(self):
        return "{}:{}-{}".format(self.model_id, self._chain_id1, self._chain_id2)

    def __eq__(self, other):
        return type(self) == type(other) and {self._chain_id1, self._chain_id2} == {other._chain_id1, other._chain_id2}

    def __hash__(self):
        return hash(tuple(sorted([self._chain_id1, self._chain_id2])))

    def __repr__(self):
        return "ProteinProteinInterfaceResidueQuery({},{})".format(self._chain_id1, self._chain_id2)

    @staticmethod
    def _residue_is_valid(residue):
        if residue.amino_acid is None:
            return False

        if residue not in residue.chain.pssm:
            _log.debug("{} not in pssm".format(residue))
            return False

        return True

    def build_graph(self):
        """ Builds the residue graph.

            Returns(deeprank graph object): the resulting graph
        """

        # get residues from the pdb

        interface_pairs = get_residue_contact_pairs(self._pdb_path, self.model_id,
                                                    self._chain_id1, self._chain_id2,
                                                    self._interface_distance_cutoff)
        if len(interface_pairs) == 0:
            raise ValueError("no interface residues found")

        interface_pairs_list = list(interface_pairs)

        model = interface_pairs_list[0].item1.chain.model
        chain1 = model.get_chain(self._chain_id1)
        chain2 = model.get_chain(self._chain_id2)

        # read the pssm
        if self._pssm_paths is not None:
            for chain in (chain1, chain2):
                pssm_path = self._pssm_paths[chain.id]

                with open(pssm_path, 'rt') as f:
                    chain.pssm = parse_pssm(f, chain)

        # separate residues by chain
        residues_from_chain1 = set([])
        residues_from_chain2 = set([])
        for residue1, residue2 in interface_pairs:

            if self._residue_is_valid(residue1):
                if residue1.chain.id == self._chain_id1:
                    residues_from_chain1.add(residue1)

                elif residue1.chain.id == self._chain_id2:
                    residues_from_chain2.add(residue1)

            if self._residue_is_valid(residue2):
                if residue2.chain.id == self._chain_id1:
                    residues_from_chain1.add(residue2)

                elif residue2.chain.id == self._chain_id2:
                    residues_from_chain2.add(residue2)

        # create the graph
        graph = Graph(self.get_query_id(), self.targets)

        # interface edges
        for pair in interface_pairs:

            residue1, residue2 = pair
            if self._residue_is_valid(residue1) and self._residue_is_valid(residue2):

                distance = get_residue_distance(residue1, residue2)

                graph.add_edge(residue1, residue2)
                graph.edges[residue1, residue2][FEATURENAME_EDGEDISTANCE] = distance
                graph.edges[residue1, residue2][FEATURENAME_EDGETYPE] = EDGETYPE_INTERFACE

        # internal edges
        for residue_set in (residues_from_chain1, residues_from_chain2):
            residue_list = list(residue_set)
            for index, residue1 in enumerate(residue_list):
                for residue2 in residue_list[index + 1:]:
                    distance = get_residue_distance(residue1, residue2)

                    if distance < self._internal_distance_cutoff:
                        graph.add_edge(residue1, residue2)
                        graph.edges[residue1, residue2][FEATURENAME_EDGEDISTANCE] = distance
                        graph.edges[residue1, residue2][FEATURENAME_EDGETYPE] = EDGETYPE_INTERNAL

        # get bsa
        pdb = pdb2sql.interface(self._pdb_path)
        try:
            bsa_calc = BSA.BSA(self._pdb_path, pdb)
            bsa_calc.get_structure()
            bsa_calc.get_contact_residue_sasa(cutoff=self._interface_distance_cutoff)
            bsa_data = bsa_calc.bsa_data
        finally:
            pdb._close()

        # get biopython features
        if self._use_biopython:
            bio_model = BioWrappers.get_bio_model(self._pdb_path)
            residue_depths = BioWrappers.get_depth_contact_res(bio_model,
                                                               [(residue.chain.id, residue.number, residue.amino_acid.three_letter_code)
                                                                for residue in graph.nodes])
            hse = BioWrappers.get_hse(bio_model)

        # add node features
        chain_codes = {chain1: 0.0, chain2: 1.0}
        for residue in graph.nodes:
            residue_key = (residue.chain.id, residue.number, residue.amino_acid.three_letter_code)
            bio_key = (residue.chain.id, residue.number)

            pssm_row = residue.get_pssm()
            pssm_value = [pssm_row.conservations[amino_acid] for amino_acid in self.amino_acid_order]

            graph.nodes[residue][FEATURENAME_CHAIN] = chain_codes[residue.chain]
            graph.nodes[residue][FEATURENAME_POSITION] = numpy.mean([atom.position for atom in residue.atoms])
            graph.nodes[residue][FEATURENAME_AMINOACID] = residue.amino_acid.onehot
            graph.nodes[residue][FEATURENAME_CHARGE] = residue.amino_acid.charge
            graph.nodes[residue][FEATURENAME_POLARITY] = residue.amino_acid.polarity.onehot
            graph.nodes[residue][FEATURENAME_BURIEDSURFACEAREA] = bsa_data[residue_key]

            if self._pssm_paths is not None:
                graph.nodes[residue][FEATURENAME_PSSM] = pssm_value
                graph.nodes[residue][FEATURENAME_CONSERVATION] = pssm_row.conservations[residue.amino_acid]
                graph.nodes[residue][FEATURENAME_INFORMATIONCONTENT] = pssm_row.information_content

            if self._use_biopython:
                graph.nodes[residue][FEATURENAME_RESIDUEDEPTH] = residue_depths[residue] if residue in residue_depths else 0.0
                graph.nodes[residue][FEATURENAME_HALFSPHEREEXPOSURE] = hse[bio_key] if bio_key in hse else (0.0, 0.0, 0.0)

        return graph

    amino_acid_order = [alanine, arginine, asparagine, aspartate, cysteine, glutamine, glutamate, glycine, histidine, isoleucine,
                        leucine, lysine, methionine, phenylalanine, proline, serine, threonine, tryptophan, tyrosine, valine]


class QueryDataset:
    "represents a collection of data queries"

    def __init__(self):
        self._queries = []

    def add(self, query):
        self._queries.append(query)

    @property
    def queries(self):
        return self._queries

    def __contains__(self, query):
        return query in self._queries

    def __iter__(self):
        return iter(self._queries)
