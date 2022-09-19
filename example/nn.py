from deeprankcore.NeuralNet import NeuralNet
from deeprankcore.DataSet import HDF5DataSet
from deeprankcore.ginet import GINet

hdf5_path = "./1ATN_residue.hdf5"

dataset = HDF5DataSet(
    root="./",
    hdf5_path=hdf5_path,
    subset=None,
    node_feature=["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
    edge_feature=["dist"],
    target="irmsd",
    clustering_method='mcl',
)

NN = NeuralNet(
    dataset,
    GINet,
    task="reg",
    batch_size=64,
    percent=[0.8, 0.2]
)

NN.train(nepoch=250, validate=False)
NN.plot_scatter()
