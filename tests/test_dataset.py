import unittest
from deeprankcore.dataset import GraphDataset, save_hdf5_keys
from deeprankcore.trainer import _divide_dataset
from torch_geometric.data.data import Data
import h5py
from deeprankcore.domain import (edgestorage as Efeat, nodestorage as Nfeat,
                                targetstorage as targets)

node_feats = [Nfeat.RESTYPE, Nfeat.POLARITY, Nfeat.BSA, Nfeat.RESDEPTH, Nfeat.HSE, Nfeat.INFOCONTENT, Nfeat.PSSM]

class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.hdf5_path = "tests/data/hdf5/1ATN_ppi.hdf5"

    def test_dataset(self):
        GraphDataset(
            hdf5_path=self.hdf5_path,
            node_features=node_feats,
            edge_features=[Efeat.DISTANCE],
            target=targets.IRMSD,
            subset=None,
        )

    def test_dataset_filter(self):
        GraphDataset(
            hdf5_path=self.hdf5_path,
            node_features=node_feats,
            edge_features=[Efeat.DISTANCE],
            target=targets.IRMSD,
            subset=None,
            target_filter={targets.IRMSD: "<10"},
        )

    def test_transform(self):

        def operator(data: Data):
            data.x = data.x / 10
            return data

        dataset = GraphDataset(
            hdf5_path=self.hdf5_path,
            node_features=node_feats,
            edge_features=[Efeat.DISTANCE],
            target=targets.IRMSD,
            subset=None,
            transform=operator
        )

        assert dataset.len() > 0
        assert dataset.get(0) is not None

    def test_multi_file_dataset(self):
        dataset = GraphDataset(
            hdf5_path=["tests/data/hdf5/train.hdf5", "tests/data/hdf5/valid.hdf5"],
            node_features=node_feats,
            edge_features=[Efeat.DISTANCE],
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
            dataset_train, dataset_val =_divide_dataset(
                dataset = GraphDataset(hdf5_path=hdf5),
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
                _divide_dataset(
                    dataset = GraphDataset(hdf5_path=hdf5),
                    val_size=t,
                )
        
        hdf5_file.close()

    def test_subset(self):
        hdf5 = h5py.File("tests/data/hdf5/train.hdf5", 'r')  # contains 44 datapoints
        hdf5_keys = list(hdf5.keys())
        n = 10
        subset = hdf5_keys[:n]

        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/train.hdf5",
            subset=subset,
        )

        assert n == len(dataset)

        hdf5.close()

    def test_target_transform(self):

        dataset = GraphDataset(
            hdf5_path = "tests/data/hdf5/train.hdf5",
            target = "BA", # continuous values
            task = 'regress',
            target_transform = True
        )

        for i in range(len(dataset)):
            assert(0 <= dataset.get(i).y <= 1)

    def test_invalid_target_transform(self):

        dataset = GraphDataset(
            hdf5_path = "tests/data/hdf5/train.hdf5",
            target = "BA", # continuous values
            task = 'classif',
            target_transform = True # only for regression
        )

        with self.assertRaises(ValueError):
            dataset.get(0)

        

if __name__ == "__main__":
    unittest.main()
