from enum import Enum
import logging

from deeprank_gnn.tools.pssm import parse_pssm
from deeprank_gnn.tools.pdb import get_residue_contact_pairs, get_residue_distance
from deeprank_gnn.models.graph import Graph
from deeprank_gnn.domain.graph import EDGETYPE_INTERNAL, EDGETYPE_INTERFACE


_log = logging.getLogger(__name__)


class Query:
    "objects of this class are created before any model is loaded"

    def __init__(self, model_id, target=None):
        self._model_id = model_id
        self._target = target

    @property
    def model_id(self):
        return self._model_id


class SingleResidueVariantQuery(Query):
    def __init__(self, model_id, chain_id, residue_number, wildtype_amino_acid, variant_amino_acid, nonbonded_distance_cutoff=10.0, target=None, insertion_code=None):
        Query.__init__(QueryType.SINGLE_RESIDUE_VARIANT, model_id, distance_cutoff, target)

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

    def __init__(self, model_id, chain_id1, chain_id2, interface_distance_cutoff=8.5, internal_distance_cutoff=3.0, target=None):
        Query.__init__(self, model_id, target)

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._interface_distance_cutoff = interface_distance_cutoff
        self._internal_distance_cutoff = internal_distance_cutoff

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

    def build_graph(self, environment):

        # get residues from the pdb
        interface_pairs = get_residue_contact_pairs(environment,
                                                    self.model_id, self._chain_id1, self._chain_id2,
                                                    self._interface_distance_cutoff)
        if len(interface_pairs) == 0:
            raise ValueError("no interface residues found")

        interface_pairs_list = list(interface_pairs)

        model = interface_pairs_list[0].item1.chain.model
        chain1 = model.get_chain(self._chain_id1)
        chain2 = model.get_chain(self._chain_id2)

        # read the pssm
        for chain in (chain1, chain2):
            pssm_path = environment.get_pssm_path(self.model_id, chain.id)
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
        graph = Graph(self.get_query_id())

        # interface edges
        for pair in interface_pairs:

            residue1, residue2 = pair
            if self._residue_is_valid(residue1) and self._residue_is_valid(residue2):

                distance = get_residue_distance(residue1, residue2)

                graph.add_edge(residue1, residue2, dist=distance, type=EDGETYPE_INTERFACE)

        # internal edges
        for residue_set in (residues_from_chain1, residues_from_chain2):
            residue_list = list(residue_set)
            for index, residue1 in enumerate(residue_list):
                for residue2 in residue_list[index + 1:]:
                    distance = get_residue_distance(residue1, residue2)

                    if distance < self._internal_distance_cutoff:
                        graph.add_edge(residue1, residue2, dist=distance, type=EDGETYPE_INTERNAL)

        return graph


class QueryDataset:
    def __init__(self):
        self._queries = []

    def add(self, query):
        self._queries.append(query)

    @property
    def queries(self):
        return self._queries

    def __contains__(self, query):
        return query in self._queries

