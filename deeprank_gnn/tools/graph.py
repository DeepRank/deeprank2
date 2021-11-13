from deeprank_gnn.models.graph import Graph


class GraphBuilder:
    def __init__(self, pdb_directory_path, pssm_directory_path):
        self._pdb_directory_path = pdb_directory_path
        self._pssm_directory_path = pssm_directory_path

    def _build_one_atomic_graph(self, query):
        structure = self._get_structure(query.model_id)

        distance_matrix = get_interatomic_distance_matrix(structure)

        graph = query.build_atomic_graph_from(distance_matrix)

        # TODO: add features

