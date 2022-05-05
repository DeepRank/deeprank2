import tempfile
import shutil
import os

import unittest

from deeprank_gnn.tools.CustomizeGraph import add_target
from deeprank_gnn.GraphGenMP import GraphHDF5
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet
from deeprank_gnn.foutnet import FoutNet
from deeprank_gnn.sGAT import sGAT


<<<<<<< HEAD
def _model_base_test( # pylint: disable=too-many-arguments
    work_directory, hdf5_path, model, task="reg", target="irmsd", plot=False
):

    NN = NeuralNet(
        hdf5_path,
        model,
        node_feature=["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
        edge_feature=["dist"],
        target=target,
        index=None,
        task=task,
        batch_size=64,
        percent=[0.8, 0.2],
        outdir=work_directory,
    )
=======
def _model_base_test(work_directory, hdf5_path, model_class,
                     node_features, edge_features,
                     task, target,
                     plot=False):

    NN = NeuralNet(hdf5_path, model_class,
                   node_feature=node_features,
                   edge_feature=edge_features,
                   target=target,
                   index=None,
                   task=task,
                   batch_size=64,
                   percent=[0.8, 0.2],
                   outdir=work_directory)
>>>>>>> main

    NN.train(nepoch=1, validate=True)

    NN.save_model("test.pth.tar")

<<<<<<< HEAD
    NeuralNet(hdf5_path, model, pretrained_model="test.pth.tar")
=======
    NN_cpy = NeuralNet(hdf5_path, model_class,
                       pretrained_model='test.pth.tar')
>>>>>>> main

    if plot:
        NN.plot_scatter()
        NN.plot_loss()
        NN.plot_acc()
        NN.plot_hit_rate()


class TestNeuralNet(unittest.TestCase):
<<<<<<< HEAD
    def setUp(self):
        f, self.hdf5_path = tempfile.mkstemp(prefix="1ATN_residue", suffix=".hdf5")
        os.close(f)

        self.work_directory = tempfile.mkdtemp()

        GraphHDF5(
            pdb_path="./tests/data/pdb/1ATN/",
            ref_path="./tests/data/ref/1ATN/",
            pssm_path="./tests/data/pssm/1ATN/",
            graph_type="residue",
            outfile=self.hdf5_path,
            nproc=1,
            tmpdir=self.work_directory,
            biopython=True,
        )

        f, target_path = tempfile.mkstemp()
        os.close(f)

        try:
            with open(target_path, "w", encoding="utf-8") as f:
                for i in range(1, 5):
                    f.write("residue-ppi-1ATN_%dw:A-B %d\n" % (i, i % 2 == 0)) # pylint: disable=consider-using-f-string

            add_target(self.hdf5_path, "bin_class", target_path)
        finally:
            os.remove(target_path)
=======

    @classmethod
    def setUpClass(class_):
        class_.work_directory = tempfile.mkdtemp()
>>>>>>> main

    @classmethod
    def tearDownClass(class_):
        shutil.rmtree(class_.work_directory)

    def test_ginet(self):
        _model_base_test(self.work_directory, "tests/hdf5/1ATN_ppi.hdf5", GINet,
                         ['type', 'polarity', 'bsa', 'depth', 'hse', 'ic', 'pssm'], ['dist'],
                         'reg', 'irmsd', plot=True)

    def test_ginet_class(self):
<<<<<<< HEAD
        _model_base_test(
            self.work_directory, self.hdf5_path, GINet, task="class", target="bin_class"
        )
=======
        _model_base_test(self.work_directory, "tests/hdf5/variants.hdf5", GINet,
                         ['size', 'polarity', 'sasa', 'ic', 'pssm'], ['dist'],
                         'class', 'bin_class')
>>>>>>> main

    def test_fout(self):
        _model_base_test(self.work_directory, "tests/hdf5/1ATN_ppi.hdf5", FoutNet,
                         ['type', 'polarity', 'bsa', 'depth', 'hse', 'ic', 'pssm'], ['dist'],
                         'reg', 'irmsd')

    def test_sgat(self):
        _model_base_test(self.work_directory, "tests/hdf5/1ATN_ppi.hdf5", sGAT,
                         ['type', 'polarity', 'bsa', 'depth', 'hse', 'ic', 'pssm'], ['dist'],
                         'reg', 'irmsd')


if __name__ == "__main__":
    unittest.main()
