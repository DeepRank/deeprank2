import unittest
import h5py
from torch_geometric.data.data import Data
from deeprankcore.dataset import GraphDataset, save_hdf5_keys
from deeprankcore.domain import targetstorage as targets


class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.hdf5_path = "tests/data/hdf5/1ATN_ppi.hdf5"

    def test_dataset(self):
        GraphDataset(
            hdf5_path=self.hdf5_path,
            target=targets.IRMSD,
        )

    def test_dataset_filter(self):
        GraphDataset(
            hdf5_path=self.hdf5_path,
            target=targets.IRMSD,
            target_filter={targets.IRMSD: "<10"},
        )

    def test_transform(self):

        def operator(data: Data):
            data.x = data.x / 10
            return data

        dataset = GraphDataset(
            hdf5_path=self.hdf5_path,
            target=targets.IRMSD,
            transform=operator
        )

        assert dataset.len() > 0
        assert dataset.get(0) is not None

    def test_multi_file_dataset(self):
        dataset = GraphDataset(
            hdf5_path=["tests/data/hdf5/train.hdf5", "tests/data/hdf5/valid.hdf5"],
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
        

if __name__ == "__main__":
    unittest.main()
