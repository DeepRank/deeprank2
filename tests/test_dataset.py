import unittest
from torch_geometric.data.data import Data
import h5py
from deeprankcore.dataset import GraphDataset, GridDataset, save_hdf5_keys
from deeprankcore.domain import (edgestorage as Efeat, nodestorage as Nfeat,
                                targetstorage as targets)

node_feats = [Nfeat.RESTYPE, Nfeat.POLARITY, Nfeat.BSA, Nfeat.RESDEPTH, Nfeat.HSE, Nfeat.INFOCONTENT, Nfeat.PSSM]

class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.hdf5_path = "tests/data/hdf5/1ATN_ppi.hdf5"

    def test_graph_dataset(self):
        dataset = GraphDataset(
            hdf5_path=self.hdf5_path,
            node_features=node_feats,
            edge_features=[Efeat.DISTANCE],
            target=targets.IRMSD,
            subset=None,
        )

        assert len(dataset) == 4
        assert dataset[0] is not None

    def test_grid_dataset_regression(self):
        dataset = GridDataset(
            hdf5_path=self.hdf5_path,
            features=[Efeat.VANDERWAALS, Efeat.ELECTROSTATIC],
            target=targets.IRMSD
        )

        assert len(dataset) == 4

        # 1 entry, 2 features with grid box dimensions
        assert dataset[0].x.shape == (1, 2, 20, 20, 20), f"got features shape {dataset[0].x.shape}"

        # 1 entry with rmsd value
        assert dataset[0].y.shape == (1,)

    def test_grid_dataset_classification(self):
        dataset = GridDataset(
            hdf5_path=self.hdf5_path,
            features=[Efeat.VANDERWAALS, Efeat.ELECTROSTATIC],
            target=targets.BINARY
        )

        assert len(dataset) == 4

        # 1 entry, 2 features with grid box dimensions
        assert dataset[0].x.shape == (1, 2, 20, 20, 20), f"got features shape {dataset[0].x.shape}"

        # 1 entry with class value
        assert dataset[0].y.shape == (1,)

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
            target = targets.BA, # continuous values --> regression
            target_transform = True
        )

        for i in range(len(dataset)):
            assert(0 <= dataset.get(i).y <= 1)

    def test_invalid_target_transform(self):

        dataset = GraphDataset(
            hdf5_path = "tests/data/hdf5/train.hdf5",
            target = targets.BINARY, # --> classificcation
            target_transform = True # only for regression
        )

        with self.assertRaises(ValueError):
            dataset.get(0)

        

if __name__ == "__main__":
    unittest.main()
