from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.DataSet import HDF5DataSet
from deeprank_gnn.ginet import GINet

database = "./1ATN_residue.hdf5"

dataset = HDF5DataSet(
    root="./",
    database=database,
    index=None,
    node_feature=["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
    edge_feature=["dist"],
    target="irmsd",
    clustering_method='mcl',
)

NN = NeuralNet(
    dataset,
    GINet,
    node_feature=["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
    edge_feature=["dist"],
    target="irmsd",
    index=None,
    task="reg",
    batch_size=64,
    percent=[0.8, 0.2],
    clustering_method='mcl'
)

NN.train(nepoch=250, validate=False)
NN.plot_scatter()
