from tempfile import mkdtemp
from shutil import rmtree
from os.path import join
import h5py
from deeprankcore.preprocess import preprocess
from typing import List, Union
from types import ModuleType
from deeprankcore.features import surfacearea
from deeprankcore.query import SingleResidueVariantResidueQuery
from deeprankcore.domain.aminoacidlist import alanine, phenylalanine
from tests._utils import PATH_TEST


def preprocess_tester(feature_modules: Union[List[ModuleType], str]):
    """
    Generic function to test preprocessing several PDB files into their feature representation HDF5 file.

    Args:
        feature_modules: list of feature modules (from .deeprankcore.feature) to be passed to preprocess.
        If "all", all available modules in deeprankcore.features are used to generate the features.
    """

    output_directory = mkdtemp()

    prefix = join(output_directory, "test-preprocess")


    try:
        count_queries = 10
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

        output_paths = preprocess(queries, prefix, 10, feature_modules)
        assert len(output_paths) > 0

        graph_names = []
        for path in output_paths:
            with h5py.File(path, "r") as f5:
                graph_names += list(f5.keys())

        for query in queries:
            query_id = query.get_query_id()
            assert query_id in graph_names, f"missing in output: {query_id}"

    finally:
        rmtree(output_directory)


def test_preprocess_single_feature():
    """
    Tests preprocessing for single feature.
    """

    preprocess_tester([surfacearea])


def test_preprocess_all_features():
    """
    Tests preprocessing for all features.
    """

    preprocess_tester("all")