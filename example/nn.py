from deeprankcore.Trainer import Trainer
from deeprankcore.DataSet import HDF5DataSet
from deeprankcore.ginet import GINet
from deeprankcore.domain.features import nodefeats as Nfeat
from deeprankcore.domain.features import edgefeats
from deeprankcore.domain import targets

hdf5_path = "./1ATN_residue.hdf5"

dataset = HDF5DataSet(
    root="./",
    hdf5_path=hdf5_path,
    subset=None,
    node_feature=[Nfeat.RESTYPE, Nfeat.POLARITY, Nfeat.BSA, Nfeat.RESDEPTH, Nfeat.HSE, Nfeat.INFOCONTENT, Nfeat.PSSM],
    edge_feature=[edgefeats.DISTANCE],
    target=targets.IRMSD,
    clustering_method='mcl',
)

trainer = Trainer(
    dataset,
    GINet,
    val_size=0.25,
    batch_size=64,
)

trainer.train(nepoch=250, validate=False)
trainer.plot_scatter()
