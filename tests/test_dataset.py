import unittest
import numpy as np
import h5py
from tempfile import mkdtemp
from shutil import rmtree
import os
from torch_geometric.loader import DataLoader
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

    def test_dataset_collates_entry_names(self):

        for dataset_name, dataset in [("GraphDataset", GraphDataset(self.hdf5_path,
                                                                    node_features=node_feats,
                                                                    edge_features=[Efeat.DISTANCE],
                                                                    target=targets.IRMSD)),
                                      ("GridDataset", GridDataset(self.hdf5_path,
                                                                  features=[Efeat.VANDERWAALS],
                                                                  target=targets.IRMSD))]:

            entry_names = []
            for batch_data in DataLoader(dataset, batch_size=2, shuffle=True):
                entry_names += batch_data.entry_names

            assert set(entry_names) == set(['residue-ppi-1ATN_1w:A-B',
                                            'residue-ppi-1ATN_2w:A-B',
                                            'residue-ppi-1ATN_3w:A-B',
                                            'residue-ppi-1ATN_4w:A-B']), f"entry names of {dataset_name} were not collated correctly"

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
            target = targets.BINARY, # --> classification
            target_transform = True # only for regression
        )

        with self.assertRaises(ValueError):
            dataset.get(0)

    def test_graph_hdf5_to_pandas(self):

        hdf5_path = "tests/data/hdf5/test.hdf5"
        dataset = GraphDataset(
            hdf5_path = hdf5_path,
            node_features='charge',
            edge_features=['distance', 'same_chain'],
            target='binary'
        )
        df = dataset.hdf5_to_pandas()
        cols = list(df.columns)
        cols.sort()
        
        assert df.shape[0] == len(dataset)
        assert df.shape[1] == 5
        assert cols == ['binary', 'charge', 'distance', 'id', 'same_chain']
        
        with h5py.File(hdf5_path, 'r') as f:
            keys = list(f.keys())
        dataset = GraphDataset(
            hdf5_path = hdf5_path,
            node_features='charge',
            edge_features=['distance', 'same_chain'],
            target='binary',
            subset=keys[2:]
        )
        df = dataset.hdf5_to_pandas()

        assert df.shape[0] == len(keys[2:])

    def test_graph_save_hist(self):

        output_directory = mkdtemp()
        fname = os.path.join(output_directory, "test.png")
        hdf5_path = "tests/data/hdf5/test.hdf5"

        dataset = GraphDataset(
            hdf5_path = hdf5_path,
            target='binary'
        )

        dataset.hdf5_to_pandas()

        dataset.save_hist(['charge', 'binary'], fname = fname)

        assert len(os.listdir(output_directory)) > 0

        rmtree(output_directory)

    def test_graph_standardize(self):

        hdf5_path = "tests/data/hdf5/train.hdf5"

        dataset = GraphDataset(
            hdf5_path = "tests/data/hdf5/train.hdf5",
            target='binary',
            standardize=True
        )

        with h5py.File(hdf5_path, 'r') as f5:
            grp = f5[list(f5.keys())[0]]

            # getting all node features values
            tensor_idx = 0
            features_dict = {}
            for feat in dataset.node_features:
                vals = grp[f"{Nfeat.NODE}/{feat}"][()]
                if vals.ndim == 1: # features with only one channel
                    arr = []
                    for entry_idx in range(len(dataset)):
                        arr.append(dataset.get(entry_idx).x[:, tensor_idx])
                    arr = np.concatenate(arr)
                    features_dict[feat] = arr
                    tensor_idx += 1
                else:
                    for ch in range(vals.shape[1]):
                        arr = []
                        for entry_idx in range(len(dataset)):
                            arr.append(dataset.get(entry_idx).x[:, tensor_idx])
                        tensor_idx += 1
                        arr = np.concatenate(arr)
                        features_dict[feat + f'_{ch}'] = arr

            for _, values in features_dict.items():

                mean = values.mean()
                dev = values.std()

                assert -0.3 < mean < 0.3
                # for one hot encoded features, it can happen that mean and std are not exactly 0 and 1
                assert 0.7 < dev < 1.5

            # getting all edge features values
            tensor_idx = 0
            features_dict = {}
            for feat in dataset.edge_features:
                vals = grp[f"{Efeat.EDGE}/{feat}"][()]
                if vals.ndim == 1: # features with only one channel
                    arr = []
                    for entry_idx in range(len(dataset)):
                        arr.append(dataset.get(entry_idx).x[:, tensor_idx])
                    arr = np.concatenate(arr)
                    features_dict[feat] = arr
                    tensor_idx += 1
                else:
                    for ch in range(vals.shape[1]):
                        arr = []
                        for entry_idx in range(len(dataset)):
                            arr.append(dataset.get(entry_idx).x[:, tensor_idx])
                        tensor_idx += 1
                        arr = np.concatenate(arr)
                        features_dict[feat + f'_{ch}'] = arr

            for _, values in features_dict.items():

                mean = values.mean()
                dev = values.std()

                assert -0.1 < mean < 0.1
                assert 0.8 < dev < 1.2


if __name__ == "__main__":
    unittest.main()
