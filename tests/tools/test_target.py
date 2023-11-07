import os
import shutil
import tempfile
import unittest
from pdb2sql import StructureSimilarity
from deeprank2.tools.target import add_target
from deeprank2.tools.target import compute_ppi_scores


class TestTools(unittest.TestCase):
    def setUp(self):
        self.pdb_path = "./tests/data/pdb/1ATN/"
        self.pssm_path = "./tests/data/pssm/1ATN/1ATN.A.pdb.pssm"
        self.ref = "./tests/data/ref/1ATN/"
        self.h5_graphs = "tests/data/hdf5/1ATN_ppi.hdf5"

    def test_add_target(self):
        f, target_path = tempfile.mkstemp(prefix="target", suffix=".lst")
        os.close(f)
        f, graph_path = tempfile.mkstemp(prefix="1ATN_ppi", suffix=".hdf5")
        os.close(f)

        try:
            target_list = ""
            for i in range(1, 11):
                target_list += f"1ATN_{i}w {i}\n"
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(target_list)
            shutil.copy(self.h5_graphs, graph_path)
            add_target(graph_path, "test_target", target_path)
        finally:
            os.remove(target_path)
            os.remove(graph_path)

    def test_compute_ppi_scores(self):
        scores = compute_ppi_scores(os.path.join(self.pdb_path, "1ATN_1w.pdb"), os.path.join(self.ref, "1ATN.pdb"))

        sim = StructureSimilarity(os.path.join(self.pdb_path, "1ATN_1w.pdb"), os.path.join(self.ref, "1ATN.pdb"), enforce_residue_matching=False)
        lrmsd = sim.compute_lrmsd_fast(method="svd")
        irmsd = sim.compute_irmsd_fast(method="svd")
        fnat = sim.compute_fnat_fast()
        dockq = sim.compute_DockQScore(fnat, lrmsd, irmsd)
        binary = irmsd < 4.0
        capri = 4
        for thr, val in zip([6.0, 4.0, 2.0, 1.0], [4, 3, 2, 1]):
            if irmsd < thr:
                capri = val

        assert scores["irmsd"] == irmsd
        assert scores["lrmsd"] == lrmsd
        assert scores["fnat"] == fnat
        assert scores["dockq"] == dockq
        assert scores["binary"] == binary
        assert scores["capri_class"] == capri

    def test_compute_ppi_scores_same_struct(self):
        scores = compute_ppi_scores(os.path.join(self.pdb_path, "1ATN_1w.pdb"), os.path.join(self.pdb_path, "1ATN_1w.pdb"))

        assert scores["irmsd"] == 0.0
        assert scores["lrmsd"] == 0.0
        assert scores["fnat"] == 1.0
        assert scores["dockq"] == 1.0
        assert scores["binary"]  # True
        assert scores["capri_class"] == 1


if __name__ == "__main__":
    unittest.main()
