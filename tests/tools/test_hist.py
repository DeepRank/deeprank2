from tempfile import mkdtemp
from shutil import rmtree
import os
import h5py
from deeprankcore.tools.transform import hdf5_to_pandas, save_hist

def test_hdf5_to_pandas():

    hdf5_path = "tests/data/hdf5/test.hdf5"
    df = hdf5_to_pandas(
        hdf5_path,
        node_features='charge',
        edge_features=['distance', 'same_chain'],
        target_features='binary')

    with h5py.File(hdf5_path, 'r') as f:
        keys = list(f.keys())

    cols = list(df.columns)
    cols.sort()
    
    assert df.shape[0] == len(keys)
    assert df.shape[1] == 5
    assert cols == ['binary', 'charge', 'distance', 'id', 'same_chain']

    df = hdf5_to_pandas(hdf5_path, subset=keys[2:])

    assert df.shape[0] == len(keys[2:])
    
def test_save_hist():

    output_directory = mkdtemp()
    fname = os.path.join(output_directory, "test.png")
    hdf5_path = "tests/data/hdf5/test.hdf5"

    df = hdf5_to_pandas(
        hdf5_path)

    save_hist(df, ['charge', 'binary'], fname = fname)

    assert len(os.listdir(output_directory)) > 0

    rmtree(output_directory)