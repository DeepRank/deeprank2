import unittest
from deeprankcore.DataSet import HDF5DataSet
from torch_geometric.data.data import Data


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


if __name__ == "__main__":
    unittest.main()
