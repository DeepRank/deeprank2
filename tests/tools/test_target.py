import os
import shutil
import tempfile
import unittest

from deeprank2.tools.target import add_target, compute_targets


class TestTools(unittest.TestCase):
    def setUp(self):
        self.pdb_path = "./tests/data/pdb/1ATN/"
        self.pssm_path = "./tests/data/pssm/1ATN/1ATN.A.pdb.pssm"
        self.ref = "./tests/data/ref/1ATN/"
        self.h5_train_ref = "tests/data/train_ref/train_data.hdf5"
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


    def test_compute_targets(self):
        compute_targets("tests/data/pdb/1ATN/1ATN_1w.pdb", "tests/data/ref/1ATN/1ATN.pdb")




if __name__ == "__main__":
    unittest.main()
