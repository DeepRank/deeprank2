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


def _model_base_test(work_directory, hdf5_path, model, task='reg', target='irmsd', plot=False):

    NN = NeuralNet(hdf5_path, model,
                   node_feature=['type', 'polarity', 'bsa',
                                 'depth', 'hse', 'ic', 'pssm'],
                   edge_feature=['dist'],
                   target=target,
                   index=None,
                   task=task,
                   batch_size=64,
                   percent=[0.8, 0.2],
                   outdir=work_directory)

    NN.train(nepoch=1, validate=True)

    NN.save_model('test.pth.tar')

    NN_cpy = NeuralNet(hdf5_path, model,
                       pretrained_model='test.pth.tar')

    if plot:
        NN.plot_scatter()
        NN.plot_loss()
        NN.plot_acc()
        NN.plot_hit_rate()


class TestNeuralNet(unittest.TestCase):

    @classmethod
    def setUpClass(class_):
        f, class_.hdf5_path = tempfile.mkstemp(prefix="1ATN_residue", suffix=".hdf5")
        os.close(f)

        class_.work_directory = tempfile.mkdtemp()

        GraphHDF5(pdb_path='./tests/data/pdb/1ATN/',
                  ref_path='./tests/data/ref/1ATN/',
                  pssm_path='./tests/data/pssm/1ATN/',
                  graph_type='residue', outfile=class_.hdf5_path,
                  nproc=4, tmpdir=class_.work_directory, biopython=True)

        f, target_path = tempfile.mkstemp()
        os.close(f)

        try:
            with open(target_path, 'w') as f:
                for i in range(1, 5):
                    f.write('residue-ppi-1ATN_%dw:A-B %d\n' % (i, i % 2 == 0))

            add_target(class_.hdf5_path, "bin_class", target_path)
        finally:
            os.remove(target_path)

    @classmethod
    def tearDownClass(class_):
        os.remove(class_.hdf5_path)
        shutil.rmtree(class_.work_directory)

    def test_ginet(self):
        _model_base_test(self.work_directory, self.hdf5_path, GINet, plot=True)

    def test_ginet_class(self):
        _model_base_test(self.work_directory, self.hdf5_path, GINet,
                         task='class', target='bin_class')

    def test_fout(self):
        _model_base_test(self.work_directory, self.hdf5_path, FoutNet)

    def test_sgat(self):
        _model_base_test(self.work_directory, self.hdf5_path, sGAT)


if __name__ == "__main__":
    unittest.main()
