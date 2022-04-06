from tempfile import mkdtemp
from shutil import rmtree
import os

import h5py

from deeprank_gnn.preprocess import PreProcessor
from deeprank_gnn.models.query import SingleResidueVariantResidueQuery
from deeprank_gnn.domain.amino_acid import alanine, phenylalanine


def test_preprocess():

    output_directory = mkdtemp()

    prefix = os.path.join(output_directory, "test-preprocess")

    preprocessor = PreProcessor(prefix)
    try:
        preprocessor.start()

        count_queries = 100
        for number in range(1, count_queries + 1):
            query = SingleResidueVariantResidueQuery("tests/data/pdb/101M/101M.pdb", "A", number, None,
                                                     alanine, phenylalanine,
                                                     pssm_paths={"A": "tests/data/pssm/101M/101M.A.pdb.pssm"},
                                                     variant_conservation=0.0, wildtype_conservation=0.0)
            preprocessor.add_query(query)

        preprocessor.wait()

        assert len(preprocessor.output_paths) > 0

        count_graphs = 0
        for path in preprocessor.output_paths:
            with h5py.File(path, 'r') as f5:
                count_graphs += len(f5.keys())

        assert count_queries == count_graphs, f"the number of hdf5 graphs doesn't match the number of queries: {count_graphs} != {count_queries}"

    finally:
        rmtree(output_directory)
