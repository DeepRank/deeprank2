from tempfile import mkdtemp
from shutil import rmtree
from os.path import basename, isfile, join
import glob
import importlib
from typing import List
import h5py
from deeprankcore.preprocess import preprocess
from deeprankcore.query import SingleResidueVariantResidueQuery
from deeprankcore.domain.aminoacidlist import alanine, phenylalanine
from tests._utils import PATH_TEST


def preprocess_tester(feature_modules: List, process_count: int, combine_files: bool):
    """
    Generic function to test preprocessing several PDB files into their feature representation HDF5 file.

    Args:
        feature_modules: list of feature modules (from .deeprankcore.feature) to be passed to preprocess
    """

    output_directory = mkdtemp()
    prefix = join(output_directory, "test-preprocess")
    count_queries = 10

    try:
        queries = []
        for number in range(1, count_queries + 1):
            query = SingleResidueVariantResidueQuery(
                str(PATH_TEST / "data/pdb/101M/101M.pdb"),
                "A",
                number,
                None,
                alanine,
                phenylalanine,
                pssm_paths={"A": str(PATH_TEST / "data/pssm/101M/101M.A.pdb.pssm")},
            )
            queries.append(query)

        output_paths = preprocess(feature_modules, queries, prefix, process_count, combine_files)
        assert len(output_paths) > 0

        graph_names = []
        for path in output_paths:
            with h5py.File(path, "r") as f5:
                graph_names += list(f5.keys())

        for query in queries:
            query_id = query.get_query_id()
            assert query_id in graph_names, f"missing in output: {query_id}"
    except Exception as e:
        print(e)

    return output_directory, output_paths


def test_preprocess_single_feature():
    """
    Tests preprocessing for generating a single feature.
    """

    imp = importlib.import_module(('deeprankcore.features.surfacearea'))
    output_directory, _ = preprocess_tester([imp], 5, False)

    rmtree(output_directory)


def test_preprocess_all_features():
    """
    Tests preprocessing for generating all features.
    """

    # copying this from feature.__init__.py
    modules = glob.glob(join('./deeprankcore/features/', "*.py"))
    modules = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

    feature_modules = []
    for m in modules:
        imp = importlib.import_module('deeprankcore.features.' + m)
        feature_modules.append(imp)

    output_directory, _ = preprocess_tester(feature_modules, 5, False)

    rmtree(output_directory)


def test_preprocess_combine_files_true():
    """
    Tests preprocessing for combining hdf5 files into one.
    """

    modules = glob.glob(join('./deeprankcore/features/', "*.py"))
    modules = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

    feature_modules = []
    for m in modules:
        imp = importlib.import_module('deeprankcore.features.' + m)
        feature_modules.append(imp)

    output_directory, output_paths = preprocess_tester(feature_modules, 5, True)
    
    assert len(output_paths) == 1

    rmtree(output_directory)


def test_preprocess_combine_files_false():
    """
    Tests preprocessing for keeping hdf5 files .
    """

    process_count = 5

    modules = glob.glob(join('./deeprankcore/features/', "*.py"))
    modules = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

    feature_modules = []
    for m in modules:
        imp = importlib.import_module('deeprankcore.features.' + m)
        feature_modules.append(imp)

    output_directory, output_paths = preprocess_tester(feature_modules, 5, False)
    
    assert len(output_paths) == 5

    rmtree(output_directory)
