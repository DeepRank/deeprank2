import unittest
from deeprankcore.DataSet import HDF5DataSet, save_hdf5_keys, _DivideDataSet
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

    def test_save_external_links(self):
        n = 2

        with h5py.File("tests/hdf5/test.hdf5", 'r') as hdf5:
            original_ids = list(hdf5.keys())
        
        save_hdf5_keys("tests/hdf5/test.hdf5", original_ids[:n], "tests/hdf5/test_resized.hdf5")

        with h5py.File("tests/hdf5/test_resized.hdf5", 'r') as hdf5:
            new_ids = list(hdf5.keys())
            assert all(isinstance(hdf5.get(key, getlink=True), h5py.ExternalLink) for key in hdf5.keys())
  
        assert len(new_ids) == n
        for new_id in new_ids:
            assert new_id in original_ids

    def test_save_hard_links(self):
        n = 2

        with h5py.File("tests/hdf5/test.hdf5", 'r') as hdf5:
            original_ids = list(hdf5.keys())
        
        save_hdf5_keys("tests/hdf5/test.hdf5", original_ids[:n], "tests/hdf5/test_resized.hdf5", hardcopy = True)

        with h5py.File("tests/hdf5/test_resized.hdf5", 'r') as hdf5:
            new_ids = list(hdf5.keys())
            assert all(isinstance(hdf5.get(key, getlink=True), h5py.HardLink) for key in hdf5.keys())
  
        assert len(new_ids) == n
        for new_id in new_ids:
            assert new_id in original_ids

    def test_trainsize(self):
        hdf5 = "tests/hdf5/train.hdf5"
        hdf5_file = h5py.File(hdf5, 'r')    # contains 44 datapoints
        n = int ( 0.75 * len(hdf5_file) )
        n_ = len(hdf5_file) - n
        test_cases = [None, 0.75, n]
        
        for t in test_cases:
            dataset_train, dataset_val =_DivideDataSet(
                dataset = HDF5DataSet(hdf5_path=hdf5),
                train_size=t,
            )

            assert len(dataset_train) == n
            assert len(dataset_val) == n_
        
    def test_invalid_trainsize(self):

        hdf5 = "tests/hdf5/train.hdf5"
        hdf5_file = h5py.File(hdf5, 'r')    # contains 44 datapoints
        n = len(hdf5_file)
        test_cases = [
            0, 0.0,     # no zeroes allowed
            -0.5, -1,   # no negative values 
            1.1, n + 1, # cannot use more than all data as input
            # 'test'      # no strings allowed
            ]
        
        for t in test_cases:
            with self.assertRaises(ValueError):
                _DivideDataSet(
                    dataset = HDF5DataSet(hdf5_path=hdf5),
                    train_size=t,
                )
        


if __name__ == "__main__":
    unittest.main()
