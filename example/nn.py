from deeprankcore.Trainer import Trainer
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

trainer = Trainer(
    dataset,
    GINet,
    val_size=0.25,
    task="reg",
    batch_size=64,
)

trainer.train(nepoch=250, validate=False)
trainer.plot_scatter()
