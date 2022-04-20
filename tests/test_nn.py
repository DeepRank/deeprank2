import shutil
import tempfile
import unittest

from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.foutnet import FoutNet
from deeprank_gnn.ginet import GINet
from deeprank_gnn.sGAT import sGAT
from tests.utils import PATH_TEST


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

    NN.train(nepoch=5, validate=True)

    NN.save_model('test.pth.tar')

    NN_cpy = NeuralNet(hdf5_path, model,
                       pretrained_model='test.pth.tar')

    if plot:
        NN.plot_scatter()
        NN.plot_loss()
        NN.plot_acc()
        NN.plot_hit_rate()


class TestNeuralNet(unittest.TestCase):

    def setUp(self):
        self.hdf5_path = str(PATH_TEST / 'data/train_ref/1ATN_train_data.hdf5')
        self.work_directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work_directory)

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
