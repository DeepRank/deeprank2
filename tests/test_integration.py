from tempfile import mkdtemp
from shutil import rmtree
import os
import h5py
from deeprankcore.preprocess import preprocess
from deeprankcore.models.query import SingleResidueVariantResidueQuery
from deeprankcore.domain.amino_acid import alanine, phenylalanine
import deeprankcore.feature.sasa
from tests.utils import PATH_TEST
from deeprankcore.DataSet import HDF5DataSet
from deeprankcore.NeuralNet import NeuralNet
from deeprankcore.ginet import GINet
from deeprankcore.models.metrics import OutputExporter
import tempfile

def test_integration():
    """
    Tests preprocessing several PDB files into their feature representation HDF5 file.

    Then uses HDF5 generated files to train and test a GINet network.

    """

    output_directory = mkdtemp()

    prefix = os.path.join(output_directory, "test-preprocess")

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

        output_paths = preprocess([deeprankcore.feature.sasa], queries, prefix, 10)
        assert len(output_paths) > 0

        graph_names = []
        for path in output_paths:
            with h5py.File(path, "r") as f5:
                graph_names += list(f5.keys())

        for query in queries:
            query_id = query.get_query_id()
            assert query_id in graph_names, f"missing in output: {query_id}"

        
        dataset = HDF5DataSet(
            database=output_paths,
            index=None,
            node_feature=["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            edge_feature=["dist"],
            target="irmsd",
            clustering_method="mcl",
        )

        nn = NeuralNet(
            dataset,
            GINet,
            task="reg",
            batch_size=64,
            percent=[0.8, 0.2],
            metrics_exporters=[OutputExporter(tempfile.mkdtemp())],
            transform_sigmoid=True,
        )   

        nn.train(nepoch=10, validate=True) 

        nn.save_model("test.pth.tar")

        NeuralNet(dataset, GINet, pretrained_model="test.pth.tar")

        assert len(os.listdir(tempfile.mkdtemp())) > 0

    finally:
        rmtree(output_directory)