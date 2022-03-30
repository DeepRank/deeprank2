import os
import tempfile
import unittest
import h5py

from deeprank_gnn.models.graph import Graph
from deeprank_gnn.tools.graph import hdf5_to_graph, graph_to_hdf5, plotly_2d, plotly_3d
from deeprank_gnn.tools.score import get_all_scores


class TestGraph(unittest.TestCase):
    def setUp(self):
        with h5py.File("tests/hdf5/1ATN_residue.hdf5", "r") as f5:
            self.graph = hdf5_to_graph(f5["1ATN_1w"])

        self.pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"
        self.reference_path = "tests/data/pdb/1ATN/1ATN_2w.pdb"

    def test_nx2h5(self):
        f, hdf5_path = tempfile.mkstemp()
        os.close(f)

        try:
            with h5py.File(hdf5_path, "w") as f5:
                graph_to_hdf5(self.graph, f5)
        finally:
            os.remove(hdf5_path)

    def test_score(self):
        scores = get_all_scores(self.pdb_path, self.reference_path)

    def test_plot_2d(self):
        plotly_2d(self.graph, "1ATN", disable_plot=True)

    def test_plot_3d(self):
        plotly_3d(self.graph, "1ATN", disable_plot=True)


if __name__ == "__main__":
    unittest.main()
