import unittest
from deeprankcore.DataSet import HDF5DataSet, save_hdf5_keys
from torch_geometric.data.data import Data
import h5py

class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.hdf5_path = "tests/hdf5/1ATN_ppi.hdf5"

    def test_dataset(self):
        HDF5DataSet(
            hdf5_path=self.hdf5_path,
            node_feature=["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            edge_feature=["dist"],
            target="irmsd",
            index=None,
        )

    def test_dataset_filter(self):
        HDF5DataSet(
            hdf5_path=self.hdf5_path,
            node_feature=["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            edge_feature=["dist"],
            target="irmsd",
            index=None,
            dict_filter={"irmsd": "<10"},
        )

    def test_transform(self):

        def operator(data: Data):
            data.x = data.x / 10
            return data

        dataset = HDF5DataSet(
            hdf5_path=self.hdf5_path,
            node_feature=["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            edge_feature=["dist"],
            target="irmsd",
            index=None,
            transform=operator
        )

        assert dataset.len() > 0
        assert dataset.get(0) is not None

    def test_multi_file_dataset(self):
        dataset = HDF5DataSet(
            hdf5_path=["tests/hdf5/train.hdf5", "tests/hdf5/valid.hdf5"],
            node_feature=["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            edge_feature=["dist"],
            target="binary"
        )

        assert dataset.len() > 0
        assert dataset.get(0) is not None

    def test_save_function(self):
        n = 2

        with h5py.File("tests/hdf5/test.hdf5", 'r') as hdf5:
            original_ids = list(hdf5.keys())
        
        save_hdf5_keys("tests/hdf5/test.hdf5", original_ids[:n], "tests/hdf5/test_resized.hdf5")

        with h5py.File("tests/hdf5/test_resized.hdf5", 'r') as hdf5:
            new_ids = list(hdf5.keys())

        assert len(new_ids) == n
        for id in new_ids:
            assert id in original_ids


if __name__ == "__main__":
    unittest.main()
