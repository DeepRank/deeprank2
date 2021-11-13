from deeprank_gnn.models.graph import Graph


class GraphBuilder:
    def __init__(self, pdb_directory_path, pssm_directory_path, distance_cutoff_nonbonded, distance_cutoff_covalent):
        self._pdb_directory_path = pdb_directory_path
        self._pssm_directory_path = pssm_directory_path
        self._distance_cutoff_nonbonded = distance_cutoff_nonbonded
        self._distance_cutoff_covalent = distance_cutoff_covalent

    def _build_one_atomic_graph(self, query):
        structure = self._get_structure(query.model_id)

        atoms = query.get_atoms_from(structure)

        
