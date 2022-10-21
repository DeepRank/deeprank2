import unittest
from deeprankcore.DataSet import HDF5DataSet, save_hdf5_keys
from deeprankcore.Trainer import _DivideDataSet
from torch_geometric.data.data import Data
import h5py
from deeprankcore.domain.features import edgefeats 
from deeprankcore.domain.features import nodefeats as Nfeat
from deeprankcore.domain import targettypes as targets

node_feats = [Nfeat.RESTYPE, Nfeat.POLARITY, Nfeat.BSA, Nfeat.RESDEPTH, Nfeat.HSE, Nfeat.INFOCONTENT, Nfeat.PSSM]

class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.hdf5_path = "tests/data/hdf5/1ATN_ppi.hdf5"

    def test_dataset(self):
        HDF5DataSet(
            hdf5_path=self.hdf5_path,
            node_feature=node_feats,
            edge_feature=[edgefeats.DISTANCE],
            target=targets.IRMSD,
            subset=None,
        )

    def test_dataset_filter(self):
        HDF5DataSet(
            hdf5_path=self.hdf5_path,
            node_feature=node_feats,
            edge_feature=[edgefeats.DISTANCE],
            target=targets.IRMSD,
            subset=None,
            dict_filter={targets.IRMSD: "<10"},
        )

    def test_transform(self):

        def operator(data: Data):
            data.x = data.x / 10
            return data

        dataset = HDF5DataSet(
            hdf5_path=self.hdf5_path,
            node_feature=node_feats,
            edge_feature=[edgefeats.DISTANCE],
            target=targets.IRMSD,
            subset=None,
            transform=operator
        )

        assert dataset.len() > 0
        assert dataset.get(0) is not None

    def test_multi_file_dataset(self):
        dataset = HDF5DataSet(
            hdf5_path=["tests/data/hdf5/train.hdf5", "tests/data/hdf5/valid.hdf5"],
            node_feature=node_feats,
            edge_feature=[edgefeats.DISTANCE],
            target=targets.BINARY
        )

        assert dataset.len() > 0
        assert dataset.get(0) is not None

    def test_save_external_links(self):
        n = 2

        with h5py.File("tests/data/hdf5/test.hdf5", 'r') as hdf5:
            original_ids = list(hdf5.keys())
        
        save_hdf5_keys("tests/data/hdf5/test.hdf5", original_ids[:n], "tests/data/hdf5/test_resized.hdf5")

        with h5py.File("tests/data/hdf5/test_resized.hdf5", 'r') as hdf5:
            new_ids = list(hdf5.keys())
            assert all(isinstance(hdf5.get(key, getlink=True), h5py.ExternalLink) for key in hdf5.keys())
  
        assert len(new_ids) == n
        for new_id in new_ids:
            assert new_id in original_ids

    def test_save_hard_links(self):
        n = 2

        with h5py.File("tests/data/hdf5/test.hdf5", 'r') as hdf5:
            original_ids = list(hdf5.keys())
        
        save_hdf5_keys("tests/data/hdf5/test.hdf5", original_ids[:n], "tests/data/hdf5/test_resized.hdf5", hardcopy = True)

        with h5py.File("tests/data/hdf5/test_resized.hdf5", 'r') as hdf5:
            new_ids = list(hdf5.keys())
            assert all(isinstance(hdf5.get(key, getlink=True), h5py.HardLink) for key in hdf5.keys())
  
        assert len(new_ids) == n
        for new_id in new_ids:
            assert new_id in original_ids

    def test_trainsize(self):
        hdf5 = "tests/data/hdf5/train.hdf5"
        hdf5_file = h5py.File(hdf5, 'r')    # contains 44 datapoints
        n_val = int ( 0.25 * len(hdf5_file) )
        n_train = len(hdf5_file) - n_val
        test_cases = [None, 0.25, n_val]
        
        for t in test_cases:
            dataset_train, dataset_val =_DivideDataSet(
                dataset = HDF5DataSet(hdf5_path=hdf5),
                val_size=t,
            )

            assert len(dataset_train) == n_train
            assert len(dataset_val) == n_val

        hdf5_file.close()
        
    def test_invalid_trainsize(self):

        hdf5 = "tests/data/hdf5/train.hdf5"
        hdf5_file = h5py.File(hdf5, 'r')    # contains 44 datapoints
        n = len(hdf5_file)
        test_cases = [
            1.0, n,     # cannot be 100% validation data
            -0.5, -1,   # no negative values 
            1.1, n + 1, # cannot use more than all data as input
            ]
        
        for t in test_cases:
            print(t)
            with self.assertRaises(ValueError):
                _DivideDataSet(
                    dataset = HDF5DataSet(hdf5_path=hdf5),
                    val_size=t,
                )
        
        hdf5_file.close()

    def test_subset(self):
        hdf5 = h5py.File("tests/data/hdf5/train.hdf5", 'r')  # contains 44 datapoints
        hdf5_keys = list(hdf5.keys())
        n = 10
        subset = hdf5_keys[:n]

        dataset = HDF5DataSet(
            hdf5_path="tests/data/hdf5/train.hdf5",
            subset=subset,
        )

        assert n == len(dataset)

        hdf5.close()
        

if __name__ == "__main__":
    unittest.main()
