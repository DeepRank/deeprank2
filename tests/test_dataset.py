import os
import unittest
import warnings
from shutil import rmtree
from tempfile import mkdtemp
from typing import List, Union

import h5py
import numpy as np
import pytest
from deeprank2.dataset import GraphDataset, GridDataset, save_hdf5_keys
from torch_geometric.loader import DataLoader

from deeprank2.domain import edgestorage as Efeat
from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain import targetstorage as targets

node_feats = [Nfeat.RESTYPE, Nfeat.POLARITY, Nfeat.BSA, Nfeat.RESDEPTH, Nfeat.HSE, Nfeat.INFOCONTENT, Nfeat.PSSM]

def _compute_features_manually( # noqa: MC0001, pylint: disable=too-many-locals
                       hdf5_path: str,
                       features_transform: dict,
                       feat: str
    ):
    # This function returns the feature specified read from the hdf5 file,
    # after applying manually features_transform dict. It returns its mean
    # and its std after having applied eventual transformations.
    # Multi-channels features are returned as an array with multiple channels.
    with h5py.File(hdf5_path, 'r') as f:
        entry_names = [entry for entry, _ in f.items()]

        mol_key = list(f.keys())[0]
        # read available node features
        available_node_features = list(f[f"{mol_key}/{Nfeat.NODE}/"].keys())
        available_node_features = [key for key in available_node_features if key[0] != '_']  # ignore metafeatures
        # read available edge features
        available_edge_features = list(f[f"{mol_key}/{Efeat.EDGE}/"].keys())
        available_edge_features = [key for key in available_edge_features if key[0] != '_']  # ignore metafeatures

        if 'all' in features_transform:
            transform = features_transform.get('all', {}).get('transform')
        else:
            transform = features_transform.get(feat, {}).get('transform')

        if feat in available_node_features:
            feat_values = [
                f[entry_name][Nfeat.NODE][feat][:]
                if f[entry_name][Nfeat.NODE][feat][()].ndim == 1
                else f[entry_name][Nfeat.NODE][feat][()] for entry_name in entry_names]
        elif feat in available_edge_features:
            feat_values = [
                f[entry_name][Efeat.EDGE][feat][:]
                if f[entry_name][Efeat.EDGE][feat][()].ndim == 1
                else f[entry_name][Efeat.EDGE][feat][()] for entry_name in entry_names]
        else:
            print(f'Feat {feat} not present in the file.')

        #apply transformation
        if transform:
            feat_values=[transform(row) for row in feat_values]

        arr = np.array(np.concatenate(feat_values))

        mean = np.round(np.nanmean(arr, axis=0), 1) if isinstance(arr[0], np.ndarray) \
                else round(np.nanmean(arr), 1)
        dev = np.round(np.nanstd(arr, axis=0), 1) if isinstance(arr[0], np.ndarray) \
                else round(np.nanstd(arr), 1)

        return arr, mean, dev

def _compute_features_with_get(
                    hdf5_path: str,
                    dataset: GraphDataset
    ):
    # This function computes features using the Dataset `get` method,
    # so as they will be seen by the network. It returns a dictionary
    # whose keys are the features' names and values are the features' values.
    # Multi-channels features are splitted into different keys
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

        # getting all edge features values
        tensor_idx = 0
        for feat in dataset.edge_features:
            vals = grp[f"{Efeat.EDGE}/{feat}"][()]
            if vals.ndim == 1: # features with only one channel
                arr = []
                for entry_idx in range(len(dataset)):
                    arr.append(dataset.get(entry_idx).edge_attr[:, tensor_idx])
                arr = np.concatenate(arr)
                features_dict[feat] = arr
                tensor_idx += 1
            else:
                for ch in range(vals.shape[1]):
                    arr = []
                    for entry_idx in range(len(dataset)):
                        arr.append(dataset.get(entry_idx).edge_attr[:, tensor_idx])
                    tensor_idx += 1
                    arr = np.concatenate(arr)
                    features_dict[feat + f'_{ch}'] = arr
    return features_dict

def _check_inherited_params(
    inherited_params: List[str],
    dataset_train: Union[GraphDataset, GridDataset],
    dataset_test: Union[GraphDataset, GridDataset],
):
    dataset_train_vars = vars(dataset_train)
    dataset_test_vars = vars(dataset_test)

    for param in inherited_params:
        assert dataset_test_vars[param] == dataset_train_vars[param]

class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.hdf5_path = "tests/data/hdf5/1ATN_ppi.hdf5"

    def test_collates_entry_names_datasets(self):

        for dataset_name, dataset in [("GraphDataset", GraphDataset(self.hdf5_path,
                                                                    node_features = node_feats,
                                                                    edge_features = [Efeat.DISTANCE],
                                                                    target = targets.IRMSD)),
                                      ("GridDataset", GridDataset(self.hdf5_path,
                                                                  features = [Efeat.VDW],
                                                                  target = targets.IRMSD))]:

            entry_names = []
            for batch_data in DataLoader(dataset, batch_size=2, shuffle=True):
                entry_names += batch_data.entry_names

            assert set(entry_names) == set(['residue-ppi-1ATN_1w:A-B',
                                            'residue-ppi-1ATN_2w:A-B',
                                            'residue-ppi-1ATN_3w:A-B',
                                            'residue-ppi-1ATN_4w:A-B']), f"entry names of {dataset_name} were not collated correctly"

    def test_datasets(self):
        dataset_graph = GraphDataset(
            hdf5_path=self.hdf5_path,
            subset=None,
            node_features=node_feats,
            edge_features=[Efeat.DISTANCE],
            target=targets.IRMSD
        )

        dataset_grid = GridDataset(
            hdf5_path = self.hdf5_path,
            subset = None,
            features = [Efeat.DISTANCE, Efeat.COVALENT, Efeat.SAMECHAIN],
            target = targets.IRMSD
        )

        assert len(dataset_graph) == 4
        assert dataset_graph[0] is not None
        assert len(dataset_grid) == 4
        assert dataset_grid[0] is not None

    def test_regression_griddataset(self):
        dataset = GridDataset(
            hdf5_path = self.hdf5_path,
            features = [Efeat.VDW, Efeat.ELEC],
            target = targets.IRMSD
        )

        assert len(dataset) == 4

        # 1 entry, 2 features with grid box dimensions
        assert dataset[0].x.shape == (1, 2, 20, 20, 20), f"got features shape {dataset[0].x.shape}"

        # 1 entry with rmsd value
        assert dataset[0].y.shape == (1,)

    def test_classification_griddataset(self):
        dataset = GridDataset(
            hdf5_path = self.hdf5_path,
            features = [Efeat.VDW, Efeat.ELEC],
            target = targets.BINARY
        )

        assert len(dataset) == 4

        # 1 entry, 2 features with grid box dimensions
        assert dataset[0].x.shape == (1, 2, 20, 20, 20), f"got features shape {dataset[0].x.shape}"

        # 1 entry with class value
        assert dataset[0].y.shape == (1,)

    def test_inherit_info_training_griddataset(self):

        dataset_train = GridDataset(
            hdf5_path = self.hdf5_path,
            features = [Efeat.VDW, Efeat.ELEC],
            target = targets.BINARY,
            target_transform = False,
            task = targets.CLASSIF,
            classes = None
        )

        dataset_test = GridDataset(
            hdf5_path = self.hdf5_path,
            train = False,
            dataset_train = dataset_train
        )

        # features, features_dict, target, target_transform, task, and classes
        # in the test should be inherited from the train
        inherited_param = ["features", "features_dict", "target", "target_transform", "task", "classes"]
        _check_inherited_params(inherited_param, dataset_train, dataset_test)

        dataset_test = GridDataset(
            hdf5_path = self.hdf5_path,
            train = False,
            dataset_train = dataset_train,
            features = [Efeat.DISTANCE, Efeat.COVALENT, Efeat.SAMECHAIN],
            target = targets.IRMSD,
            target_transform = True,
            task = targets.REGRESS,
            classes = None
        )

        # features, features_dict, target, target_transform, task, and classes
        # in the test should be inherited from the train
        _check_inherited_params(inherited_param, dataset_train, dataset_test)

    def test_filter_graphdataset(self):
        GraphDataset(
            hdf5_path=self.hdf5_path,
            subset=None,
            node_features=node_feats,
            edge_features=[Efeat.DISTANCE],
            target=targets.IRMSD,
            target_filter={targets.IRMSD: "<10"}
        )

    def test_multi_file_graphdataset(self):
        dataset = GraphDataset(
            hdf5_path=["tests/data/hdf5/train.hdf5", "tests/data/hdf5/valid.hdf5"],
            node_features=node_feats,
            edge_features=[Efeat.DISTANCE],
            target=targets.BINARY
        )

        assert dataset.len() > 0
        assert dataset.get(0) is not None

    def test_save_external_links_graphdataset(self):
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

    def test_save_hard_links_graphdataset(self):
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

    def test_subset_graphdataset(self):
        hdf5 = h5py.File("tests/data/hdf5/train.hdf5", 'r')  # contains 44 datapoints
        hdf5_keys = list(hdf5.keys())
        n = 10
        subset = hdf5_keys[:n]

        dataset = GraphDataset(
            hdf5_path = "tests/data/hdf5/train.hdf5",
            subset = subset,
        )

        assert n == len(dataset)

        hdf5.close()

    def test_target_transform_graphdataset(self):

        dataset = GraphDataset(
            hdf5_path = "tests/data/hdf5/train.hdf5",
            target = 'BA', # continuous values --> regression
            task = targets.REGRESS,
            target_transform = True
        )

        for i in range(len(dataset)):
            assert(0 <= dataset.get(i).y <= 1)

    def test_invalid_target_transform_graphdataset(self):

        dataset = GraphDataset(
            hdf5_path = "tests/data/hdf5/train.hdf5",
            target = targets.BINARY, # --> classification
            target_transform = True # only for regression
        )

        with self.assertRaises(ValueError):
            dataset.get(0)

    def test_size_graphdataset(self):
        hdf5_paths = ["tests/data/hdf5/train.hdf5", "tests/data/hdf5/valid.hdf5", "tests/data/hdf5/test.hdf5"]
        dataset = GraphDataset(
            hdf5_path = hdf5_paths,
            node_features = node_feats,
            edge_features = [Efeat.DISTANCE],
            target = targets.BINARY
        )
        n = 0
        for hdf5 in hdf5_paths:
            with h5py.File(hdf5, 'r') as hdf5_r:
                n += len(hdf5_r.keys())
        assert len(dataset) == n, f"total data points got was {len(dataset)}"

    def test_hdf5_to_pandas_graphdataset(self):

        hdf5_path = "tests/data/hdf5/train.hdf5"
        dataset = GraphDataset(
            hdf5_path = hdf5_path,
            node_features = 'charge',
            edge_features = ['distance', 'same_chain'],
            target = 'binary'
        )
        dataset.hdf5_to_pandas()
        cols = list(dataset.df.columns)
        cols.sort()

        # assert dataset and df shapes
        assert dataset.df.shape[0] == len(dataset)
        assert dataset.df.shape[1] == 5
        assert cols == ['binary', 'charge', 'distance', 'id', 'same_chain']

        # assert dataset and df values
        with h5py.File(hdf5_path, 'r') as f5:

            # getting nodes values with get()
            tensor_idx = 0
            features_dict = {}
            for feat in dataset.node_features:
                vals = f5[list(f5.keys())[0]][f"{Nfeat.NODE}/{feat}"][()]
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

            for feat, values in features_dict.items():
                assert np.allclose(values, np.concatenate(dataset.df[feat].values))

            # getting edges values with get()
            tensor_idx = 0
            features_dict = {}
            for feat in dataset.edge_features:
                vals = f5[list(f5.keys())[0]][f"{Efeat.EDGE}/{feat}"][()]
                if vals.ndim == 1: # features with only one channel
                    arr = []
                    for entry_idx in range(len(dataset)):
                        arr.append(dataset.get(entry_idx).edge_attr[:, tensor_idx])
                    arr = np.concatenate(arr)
                    features_dict[feat] = arr
                    tensor_idx += 1
                else:
                    for ch in range(vals.shape[1]):
                        arr = []
                        for entry_idx in range(len(dataset)):
                            arr.append(dataset.get(entry_idx).edge_attr[:, tensor_idx])
                        tensor_idx += 1
                        arr = np.concatenate(arr)
                        features_dict[feat + f'_{ch}'] = arr

            for feat, values in features_dict.items():
                # edge_attr contains stacked edges (doubled) so we test on mean and std
                assert np.float32(round(values.mean(), 2)) == np.float32(round(np.concatenate(dataset.df[feat].values).mean(), 2))
                assert np.float32(round(values.std(), 2)) == np.float32(round(np.concatenate(dataset.df[feat].values).std(), 2))

        # assert dataset and df shapes in subset case
        with h5py.File(hdf5_path, 'r') as f:
            keys = list(f.keys())

        dataset = GraphDataset(
            hdf5_path = hdf5_path,
            node_features = 'charge',
            edge_features = ['distance', 'same_chain'],
            target = 'binary',
            subset = keys[2:]
        )
        dataset.hdf5_to_pandas()

        assert dataset.df.shape[0] == len(keys[2:])

    def test_save_hist_graphdataset(self):

        output_directory = mkdtemp()
        fname = os.path.join(output_directory, "test.png")
        hdf5_path = "tests/data/hdf5/test.hdf5"

        dataset = GraphDataset(
            hdf5_path = hdf5_path,
            target = 'binary'
        )

        with self.assertRaises(ValueError):
            dataset.save_hist(['non existing feature'], fname = fname)

        dataset.save_hist(['charge', 'binary'], fname = fname)

        assert len(os.listdir(output_directory)) > 0

        rmtree(output_directory)

    def test_logic_train_graphdataset(self):# noqa: MC0001, pylint: disable=too-many-locals
        hdf5_path = "tests/data/hdf5/train.hdf5"

        # without specifying features_transform in training set
        dataset_train = GraphDataset(
            hdf5_path = hdf5_path,
            target = 'binary'
        )

        dataset_test = GraphDataset(
            hdf5_path = hdf5_path,
            target = 'binary',
            train = False,
            dataset_train = dataset_train
        )
        # mean and devs should be None
        assert dataset_train.means == dataset_test.means
        assert dataset_train.devs == dataset_test.devs
        assert dataset_train.means is None
        assert dataset_train.devs is None

        # raise error if dataset_train is not provided
        with self.assertRaises(TypeError):
            GraphDataset(
                hdf5_path = hdf5_path,
                target = 'binary',
                train = False
            )

        # raise error if dataset_train is of the wrong type
        with self.assertRaises(TypeError):

            dataset_train = GridDataset(
                hdf5_path = "tests/data/hdf5/1ATN_ppi.hdf5",
                target = 'binary'
            )

            GraphDataset(
                hdf5_path = hdf5_path,
                train = False,
                dataset_train = dataset_train,
                target = 'binary',
            )

    def test_only_transform_graphdataset(self):# noqa: MC0001, pylint: disable=too-many-locals
        # define a features_transform dict for only transformations,
        # including node (bsa) and edge features (electrostatic),
        # a multi-channel feature (hse) and a case with transform equals to None (sasa)

        hdf5_path = "tests/data/hdf5/train.hdf5"
        features_transform = {'bsa': {'transform': lambda t: np.log(t+10)},
                        'electrostatic': {'transform': lambda t:np.cbrt(t)}, # pylint: disable=unnecessary-lambda
                        'sasa': {'transform': None},
                        'hse': {'transform': lambda t: np.log(t+10)}
                        }

        # dataset that has the transformations applied using features_transform dict
        transf_dataset = GraphDataset(
            hdf5_path = hdf5_path,
            features_transform = features_transform,
            target = 'binary',
        )

        # dataset with no transformations applied
        dataset = GraphDataset(
            hdf5_path = hdf5_path,
            target = 'binary',
        )

        # transformed features
        transf_features_dict = _compute_features_with_get(hdf5_path, transf_dataset)
        # not transformed features
        features_dict = _compute_features_with_get(hdf5_path, dataset)

        features = dataset.node_features + dataset.edge_features
        checked_features = []
        for transf_feat_key, transf_feat_value in transf_features_dict.items():
            # verify that the transformed feature is not all nans
            assert not np.isnan(transf_feat_value).all()
            orig_feat = None
            for feat in features:
                if feat in transf_feat_key:
                    orig_feat = feat
                    break

            if orig_feat and (orig_feat in features_transform) and (orig_feat not in checked_features):
                checked_features.append(orig_feat)
                transform = features_transform.get(orig_feat, {}).get('transform')
                arr, _, _ = _compute_features_manually(hdf5_path, features_transform, orig_feat)
                if arr.ndim == 1:
                    # checking that the mean and the std are the same in both the feature computed through
                    # the get method and the feature computed manually
                    assert np.allclose(np.nanmean(transf_feat_value), np.nanmean(arr))
                    assert np.allclose(np.nanstd(transf_feat_value), np.nanstd(arr))
                    if transform:
                        # check that the feature mean and std are different in transf_dataset and dataset
                        assert not np.allclose(np.nanmean(transf_feat_value), np.nanmean(features_dict.get(transf_feat_key)))
                        assert not np.allclose(np.nanstd(transf_feat_value), np.nanstd(features_dict.get(transf_feat_key)))
                    else:
                        # check that the feature mean and std are the same in transf_dataset and dataset, because
                        # no transformation should be applied
                        assert np.allclose(np.nanmean(transf_feat_value), np.nanmean(features_dict.get(transf_feat_key)))
                        assert np.allclose(np.nanstd(transf_feat_value), np.nanstd(features_dict.get(transf_feat_key)))
                else:
                    for i in range(arr.shape[1]):
                        # checking that the mean and the std are the same in both the feature computed through
                        # the get method and the feature computed manually
                        assert np.allclose(np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')), np.nanmean(arr[:, i]))
                        assert np.allclose(np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')), np.nanstd(arr[:, i]))
                        if transform:
                            # check that the feature mean and std are different in transf_dataset and dataset
                            assert not np.allclose(
                                np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')),
                                np.nanmean(features_dict.get(orig_feat + f'_{i}')))
                            assert not np.allclose(
                                np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')),
                                np.nanstd(features_dict.get(orig_feat + f'_{i}')))
                        else:
                            # check that the feature mean and std are the same in transf_dataset and dataset, because
                            # no transformation should be applied
                            assert np.allclose(
                                np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')),
                                np.nanmean(features_dict.get(orig_feat + f'_{i}')))
                            assert np.allclose(
                                np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')),
                                np.nanstd(features_dict.get(orig_feat + f'_{i}')))

        assert (sorted(checked_features) == sorted(list(features_transform.keys()))) and (len(checked_features) == len(features_transform.keys()))

    def test_only_transform_all_graphdataset(self):# noqa: MC0001, pylint: disable=too-many-locals
        # define a features_transform dict for only transformations for `all` features

        hdf5_path = "tests/data/hdf5/train.hdf5"
        features_transform = {'all': {'transform': lambda t: np.log(abs(t)+.01)}}

        # dataset that has the transformations applied using features_transform dict
        transf_dataset = GraphDataset(
            hdf5_path = hdf5_path,
            features_transform = features_transform,
            target = 'binary',
        )

        # dataset with no transformations applied
        dataset = GraphDataset(
            hdf5_path = hdf5_path,
            target = 'binary',
        )

        # transformed features
        transf_features_dict = _compute_features_with_get(hdf5_path, transf_dataset)
        # not transformed features
        features_dict = _compute_features_with_get(hdf5_path, dataset)

        features = dataset.node_features + dataset.edge_features
        checked_features = []
        for transf_feat_key, transf_feat_value in transf_features_dict.items():
            # verify that the transformed feature is not all nans
            assert not np.isnan(transf_feat_value).all()
            orig_feat = None
            for feat in features:
                if feat in transf_feat_key:
                    orig_feat = feat
                    break

            if orig_feat and (orig_feat not in checked_features):
                checked_features.append(orig_feat)
                arr, _, _ = _compute_features_manually(hdf5_path, features_transform, orig_feat)
                if arr.ndim == 1:
                    # checking that the mean and the std are the same in both the feature computed through
                    # the get method and the feature computed manually
                    assert np.allclose(np.nanmean(transf_feat_value), np.nanmean(arr))
                    assert np.allclose(np.nanstd(transf_feat_value), np.nanstd(arr))
                    # check that the feature mean and std are different in transf_dataset and dataset
                    assert not np.allclose(np.nanmean(transf_feat_value), np.nanmean(features_dict.get(transf_feat_key)))
                    assert not np.allclose(np.nanstd(transf_feat_value), np.nanstd(features_dict.get(transf_feat_key)))
                else:
                    for i in range(arr.shape[1]):
                        # checking that the mean and the std are the same in both the feature computed through
                        # the get method and the feature computed manually
                        assert np.allclose(np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')), np.nanmean(arr[:, i]))
                        assert np.allclose(np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')), np.nanstd(arr[:, i]))
                        assert not np.allclose(np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')), np.nanmean(features_dict.get(orig_feat + f'_{i}')))
                        # check that the feature mean and std are different in transf_dataset and dataset
                        assert not np.allclose(np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')), np.nanstd(features_dict.get(orig_feat + f'_{i}')))

        assert (sorted(checked_features) == sorted(features)) and (len(checked_features) == len(features))

    def test_only_standardize_graphdataset(self): # pylint: disable=too-many-locals
        # define a features_transform dict for only standardization,
        # including node (bsa) and edge features (electrostatic),
        # a multi-channel feature (hse) and a case with standardize False (sasa)

        hdf5_path = "tests/data/hdf5/train.hdf5"
        features_transform = {
            'bsa': {'standardize': True},
            'hse': {'standardize': True},
            'electrostatic': {'standardize': True},
            'sasa': {'standardize': False}}

        transf_dataset = GraphDataset(
            hdf5_path = hdf5_path,
            features_transform = features_transform,
            target = 'binary'
        )

        dataset = GraphDataset(
            hdf5_path = hdf5_path,
            target = 'binary'
        )

        # standardized features
        transf_features_dict = _compute_features_with_get(hdf5_path, transf_dataset)
        # not standardized features
        features_dict = _compute_features_with_get(hdf5_path, dataset)

        features = dataset.node_features + dataset.edge_features
        checked_features = []
        for transf_feat_key, transf_feat_value in transf_features_dict.items():
            # verify that the transformed feature is not all nans
            assert not np.isnan(transf_feat_value).all()
            orig_feat = None
            for feat in features:
                if feat in transf_feat_key:
                    orig_feat = feat
                    break

            if orig_feat and (orig_feat in features_transform) and (orig_feat not in checked_features):
                checked_features.append(orig_feat)
                standardize = features_transform.get(orig_feat, {}).get('standardize')
                arr, mean, dev = _compute_features_manually(hdf5_path, features_transform, orig_feat)
                if standardize:
                    # standardize manually
                    arr = (arr-mean)/dev
                if arr.ndim == 1:
                    # checking that the mean and the std are the same in both the feature computed through
                    # the get method and the feature computed manually
                    assert np.allclose(np.nanmean(transf_feat_value), np.nanmean(arr))
                    assert np.allclose(np.nanstd(transf_feat_value), np.nanstd(arr))
                    if standardize:
                        # check that the feature mean and std are different in transf_dataset and dataset
                        assert not np.allclose(np.nanmean(transf_feat_value), np.nanmean(features_dict.get(transf_feat_key)))
                        assert not np.allclose(np.nanstd(transf_feat_value), np.nanstd(features_dict.get(transf_feat_key)))
                    else:
                        # check that the feature mean and std are the same in transf_dataset and dataset, because
                        # no transformation should be applied
                        assert np.allclose(np.nanmean(transf_feat_value), np.nanmean(features_dict.get(transf_feat_key)))
                        assert np.allclose(np.nanstd(transf_feat_value), np.nanstd(features_dict.get(transf_feat_key)))
                else:
                    for i in range(arr.shape[1]):
                        # checking that the mean and the std are the same in both the feature computed through
                        # the get method and the feature computed manually
                        assert np.allclose(np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')), np.nanmean(arr[:, i]))
                        assert np.allclose(np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')), np.nanstd(arr[:, i]))
                        if standardize:
                            # check that the feature mean and std are different in transf_dataset and dataset
                            assert not np.allclose(
                                np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')),
                                np.nanmean(features_dict.get(orig_feat + f'_{i}')))
                            assert not np.allclose(
                                np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')),
                                np.nanstd(features_dict.get(orig_feat + f'_{i}')))
                        else:
                            # check that the feature mean and std are the same in transf_dataset and dataset, because
                            # no standardization should be applied
                            assert np.allclose(np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')), np.nanmean(features_dict.get(orig_feat + f'_{i}')))
                            assert np.allclose(np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')), np.nanstd(features_dict.get(orig_feat + f'_{i}')))

        assert (sorted(checked_features) == sorted(list(features_transform.keys()))) and (len(checked_features) == len(features_transform.keys()))

    def test_only_standardize_all_graphdataset(self): # pylint: disable=too-many-locals
        # define a features_transform dict for only standardization for `all` features
        hdf5_path = "tests/data/hdf5/train.hdf5"
        features_transform = {
            'all': {'standardize': True}}

        # dataset that has the standardization applied using features_transform dict
        transf_dataset = GraphDataset(
            hdf5_path = hdf5_path,
            features_transform = features_transform,
            target = 'binary'
        )

        # dataset with no standardization applied
        dataset = GraphDataset(
            hdf5_path = hdf5_path,
            target = 'binary'
        )

        # standardized features
        transf_features_dict = _compute_features_with_get(hdf5_path, transf_dataset)
        # not standardized features
        features_dict = _compute_features_with_get(hdf5_path, dataset)

        features = dataset.node_features + dataset.edge_features
        checked_features = []
        for transf_feat_key, transf_feat_value in transf_features_dict.items():
            # verify that the transformed feature is not all nans
            assert not np.isnan(transf_feat_value).all()
            orig_feat = None
            for feat in features:
                if feat in transf_feat_key:
                    orig_feat = feat
                    break

            if orig_feat and (orig_feat not in checked_features):
                checked_features.append(orig_feat)
                arr, mean, dev = _compute_features_manually(hdf5_path, features_transform, orig_feat)
                # standardize manually
                arr = (arr-mean)/dev
                if arr.ndim == 1:
                    # checking that the mean and the std are the same in both the feature computed through
                    # the get method and the feature computed manually
                    assert np.allclose(np.nanmean(transf_feat_value), np.nanmean(arr))
                    assert np.allclose(np.nanstd(transf_feat_value), np.nanstd(arr))
                    # check that the feature mean and std are different in transf_dataset and dataset
                    assert not np.allclose(np.nanmean(transf_feat_value), np.nanmean(features_dict.get(transf_feat_key)))
                    assert not np.allclose(np.nanstd(transf_feat_value), np.nanstd(features_dict.get(transf_feat_key)))
                else:
                    for i in range(arr.shape[1]):
                        # checking that the mean and the std are the same in both the feature computed through
                        # the get method and the feature computed manually
                        assert np.allclose(np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')), np.nanmean(arr[:,i]))
                        assert np.allclose(np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')), np.nanstd(arr[:,i]))
                        # check that the feature mean and std are different in transf_dataset and dataset
                        assert not np.allclose(np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')), np.nanmean(features_dict.get(orig_feat + f'_{i}')))
                        assert not np.allclose(np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')), np.nanstd(features_dict.get(orig_feat + f'_{i}')))

        assert (sorted(checked_features) == sorted(features)) and (len(checked_features) == len(features))

    def test_transform_standardize_graphdataset(self):# noqa: MC0001, pylint: disable=too-many-locals
        # define a features_transform dict for both transformations and standardization,
        # including node (bsa) and edge features (electrostatic),
        # a multi-channel feature (hse)

        hdf5_path = "tests/data/hdf5/train.hdf5"
        features_transform = {'bsa': {'transform': lambda t: np.log(t+10), 'standardize': True},
                        'electrostatic': {'transform': lambda t:np.cbrt(t), 'standardize': True}, # pylint: disable=unnecessary-lambda
                        'sasa': {'transform': None, 'standardize': False},
                        'hse': {'transform': lambda t: np.log(t+10), 'standardize': False}
                        }

        # dataset that has the transformations applied using features_transform dict
        transf_dataset = GraphDataset(
            hdf5_path = hdf5_path,
            features_transform = features_transform,
            target = 'binary'
        )

        # dataset with no transformations applied
        dataset = GraphDataset(
            hdf5_path = hdf5_path,
            target = 'binary',
        )

        # transformed features
        transf_features_dict = _compute_features_with_get(hdf5_path, transf_dataset)
        # not transformed features
        features_dict = _compute_features_with_get(hdf5_path, dataset)

        features = dataset.node_features + dataset.edge_features
        checked_features = []
        for transf_feat_key, transf_feat_value in transf_features_dict.items():
            # verify that the transformed feature is not all nans
            assert not np.isnan(transf_feat_value).all()
            orig_feat = None
            for feat in features:
                if feat in transf_feat_key:
                    orig_feat = feat
                    break

            if orig_feat and (orig_feat in features_transform) and (orig_feat not in checked_features):
                checked_features.append(orig_feat)
                transform = features_transform.get(orig_feat, {}).get('transform')
                standardize = features_transform.get(orig_feat, {}).get('standardize')
                arr, mean, dev = _compute_features_manually(hdf5_path, features_transform, orig_feat)
                if standardize:
                    # standardize manually
                    arr = (arr-mean)/dev
                if arr.ndim == 1:
                    # checking that the mean and the std are the same in both the feature computed through
                    # the get method and the feature computed manually
                    assert np.allclose(np.nanmean(transf_feat_value), np.nanmean(arr))
                    assert np.allclose(np.nanstd(transf_feat_value), np.nanstd(arr))
                    if transform or standardize:
                        # check that the feature mean and std are different in transf_dataset and dataset
                        assert not np.allclose(np.nanmean(transf_feat_value), np.nanmean(features_dict.get(transf_feat_key)))
                        assert not np.allclose(np.nanstd(transf_feat_value), np.nanstd(features_dict.get(transf_feat_key)))
                    else:
                        # check that the feature mean and std are the same in transf_dataset and dataset
                        assert np.allclose(np.nanmean(transf_feat_value), np.nanmean(features_dict.get(transf_feat_key)))
                        assert np.allclose(np.nanstd(transf_feat_value), np.nanstd(features_dict.get(transf_feat_key)))
                else:
                    for i in range(arr.shape[1]):
                        # checking that the mean and the std are the same in both the feature computed through
                        # the get method and the feature computed manually
                        assert np.allclose(np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')), np.nanmean(arr[:, i]))
                        assert np.allclose(np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')), np.nanstd(arr[:, i]))
                        if transform or standardize:
                            # check that the feature mean and std are different in transf_dataset and dataset
                            assert not np.allclose(
                                np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')),
                                np.nanmean(features_dict.get(orig_feat + f'_{i}')))
                            assert not np.allclose(
                                np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')),
                                np.nanstd(features_dict.get(orig_feat + f'_{i}')))
                        else:
                            # check that the feature mean and std are the same in transf_dataset and dataset, because
                            assert np.allclose(
                                np.nanmean(transf_features_dict.get(orig_feat + f'_{i}')),
                                np.nanmean(features_dict.get(orig_feat + f'_{i}')))
                            assert np.allclose(
                                np.nanstd(transf_features_dict.get(orig_feat + f'_{i}')),
                                np.nanstd(features_dict.get(orig_feat + f'_{i}')))

        assert (sorted(checked_features) == sorted(list(features_transform.keys()))) and (len(checked_features) == len(features_transform.keys()))

    def test_features_transform_logic_graphdataset(self):

        hdf5_path = "tests/data/hdf5/train.hdf5"
        features_transform = {'all': {'transform': lambda t:np.cbrt(t), 'standardize': True}} # pylint: disable=unnecessary-lambda
        other_feature_transform = {'all': {'transform': None, 'standardize': False}}

        dataset_train = GraphDataset(
            hdf5_path = hdf5_path,
            features_transform = features_transform,
            target = 'binary'
        )

        dataset_test = GraphDataset(
            hdf5_path = hdf5_path,
            train = False,
            dataset_train = dataset_train,
            target = 'binary'
        )

        # features_transform in the test should be the same as in the train
        assert dataset_train.features_transform == dataset_test.features_transform
        assert dataset_train.means == dataset_test.means
        assert dataset_train.devs == dataset_test.devs
        # features_transform contains standardize True, so means and devs should be computed
        assert dataset_train.means is not None
        assert dataset_train.devs is not None

        dataset_test = GraphDataset(
            hdf5_path = hdf5_path,
            train = False,
            dataset_train = dataset_train,
            features_transform = other_feature_transform,
            target = 'binary'
        )

        # features_transform setted in the testset should be ignored
        assert dataset_train.features_transform == dataset_test.features_transform
        assert dataset_train.means == dataset_test.means
        assert dataset_train.devs == dataset_test.devs

    def test_invalid_value_features_transform(self):

        hdf5_path = "tests/data/hdf5/train.hdf5"
        features_transform = {'all': {'transform': lambda t: np.log(t+10), 'standardize': True}}

        transf_dataset = GraphDataset(
            hdf5_path = hdf5_path,
            target = 'binary',
            features_transform = features_transform
        )
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'divide by zero encountered in divide')
                _compute_features_with_get(hdf5_path, transf_dataset)

    def test_inherit_info_training_graphdataset(self):
        hdf5_path = "tests/data/hdf5/train.hdf5"
        feature_transform = {'all': {'transform': None, 'standardize': True}}

        dataset_train = GraphDataset(
            hdf5_path = hdf5_path,
            node_features = ['bsa', 'hb_acceptors', 'hb_donors'],
            edge_features = ['covalent', 'distance'],
            features_transform = feature_transform,
            target = 'binary',
            target_transform = False,
            task = 'classif',
            classes = None
        )

        dataset_test = GraphDataset(
            hdf5_path = hdf5_path,
            train = False,
            dataset_train = dataset_train,
        )

        # node_features, edge_features, features_dict, feature_transform, target, target_transform, task, and classes
        # in the test should be inherited from the train
        inherited_param = ["node_features", "edge_features", "features_dict", "features_transform", "target", "target_transform", "task", "classes"]
        _check_inherited_params(inherited_param, dataset_train, dataset_test)

        dataset_test = GraphDataset(
            hdf5_path = hdf5_path,
            train = False,
            dataset_train = dataset_train,
            node_features = "all",
            edge_features = "all",
            features_transform = None,
            target = 'BA',
            target_transform = True,
            task = "regress",
            classes = None
        )

        # node_features, edge_features, features_dict, feature_transform, target, target_transform, task, and classes
        # in the test should be inherited from the train
        _check_inherited_params(inherited_param, dataset_train, dataset_test)

    def test_incompatible_dataset_train_type(self):
        dataset_train = GraphDataset(
            hdf5_path = "tests/data/hdf5/test.hdf5",
            edge_features = [Efeat.DISTANCE, Efeat.COVALENT],
            target = targets.BINARY
        )

        # Raise error when val dataset don't have the same data type as train dataset.
        with pytest.raises(TypeError):
            GridDataset(
                hdf5_path = "tests/data/hdf5/1ATN_ppi.hdf5",
                train = False,
                dataset_train = dataset_train
            )


if __name__ == "__main__":
    unittest.main()
