from enum import Enum

from deeprank_gnn.tools.pdb import get_structure


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


class ProteinProteinInterfaceQuery(Query):
    def __init__(self, model_id, chain_id1, chain_id2, distance_cutoff=8.5, target=None):
        Query.__init__(QueryType.PROTEIN_PROTEIN_INTERFACE, model_id, distance_cutoff, target)

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

    def __eq__(self, other):
        return type(self) == type(other) and {self._chain_id1, self._chain_id2} == {other._chain_id1, other._chain_id2}

    def __hash__(self):
        return hash(tuple(sorted([self._chain_id1, self._chain_id2])))

    def __repr__(self):
        return "ProteinProteinInterfaceQuery({},{})".format(self._chain_id1, self._chain_id2)

    def build_residue_graph(self, environment):

        # read the pdb
        pdb_path = environment.get_pdb_path(self.model_id)
        pdb = pdb2sql.interface(pdb_path)
        try:
            residue_contact_pairs = get_residue_contact_pairs(pdb, self.residue_distance_cutoff)
        finally:
            pdb._close()

        # read the pssm
        pssm = Pssm()
        for chain_id, pssm_path in environment.get_pssm_paths(self.model_id).items():
            pssm.update(parse_pssm(chain_id, pssm_path))

        # filter
        valid_nodes = ProteinProteinInterfaceQuery._filter_valid_nodes(residue_contact_pairs, pssm)

        # calculate distances
        distances = get_residue_distances(environment.device, residue_contact_pairs)

        # create the graph
        graph = Graph(self.get_query_id())

        for pair in residue_contact_pairs:
            residue1, residue2 = pair
            if residue1 in valid_nodes and residue2 in valid_nodes:
                distance = distances[pair]

                if ProteinProteinInterfaceQuery._is_internal_edge(residue1, residue2):

                    graph.add_edge(residue1, residue2, dist=distance, type=EDGETYPE_INTERNAL)
                else:
                    graph.add_edge(residue1, residue2, dist=distance, type=EDGETYPE_INTERFACE)

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

