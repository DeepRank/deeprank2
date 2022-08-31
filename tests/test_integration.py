from tempfile import mkdtemp
from shutil import rmtree
import os
import h5py
from deeprankcore.preprocess import preprocess
from deeprankcore.models.query import ProteinProteinInterfaceResidueQuery
from deeprankcore.feature import amino_acid, atomic_contact, biopython, bsa, pssm, sasa 
from tests.utils import PATH_TEST
from deeprankcore.DataSet import HDF5DataSet
from deeprankcore.NeuralNet import NeuralNet
from deeprankcore.ginet import GINet
from deeprankcore.models.metrics import OutputExporter
from deeprankcore.tools.score import get_all_scores
import tempfile

def test_integration(): # pylint: disable=too-many-locals
    """
    Tests preprocessing several PDB files into their features representation HDF5 file.

    Then uses HDF5 generated files to train and test a GINet network.

    """

    pdb_path = str(PATH_TEST / "data/pdb/1ATN/1ATN_1w.pdb")
    ref_path = str(PATH_TEST / "data/ref/1ATN/1ATN.pdb")
    pssm_path1 = str(PATH_TEST / "data/pssm/1ATN/1ATN.A.pdb.pssm")
    pssm_path2 = str(PATH_TEST / "data/pssm/1ATN/1ATN.B.pdb.pssm")
    chain_id1 = "A"
    chain_id2 = "B"

    output_directory = mkdtemp()
    metrics_directory = tempfile.mkdtemp()

    prefix = os.path.join(output_directory, "test-preprocess")

    feature_modules = [amino_acid, atomic_contact, biopython, bsa, pssm, sasa]

    try:

        targets = get_all_scores(pdb_path, ref_path)

        count_queries = 10
        queries = []
        for _ in range(1, count_queries + 1):
            query = ProteinProteinInterfaceResidueQuery(
                pdb_path,
                chain_id1,
                chain_id2,
                pssm_paths={chain_id1: pssm_path1, chain_id2: pssm_path2},
                targets = targets
            )
            queries.append(query)

        output_paths = preprocess(feature_modules, queries, prefix, 10)
        assert len(output_paths) > 0

        graph_names = []
        for path in output_paths:
            with h5py.File(path, "r") as f5:
                graph_names += list(f5.keys())

        for query in queries:
            query_id = query.get_query_id()
            assert query_id in graph_names, f"missing in output: {query_id}"

        
        dataset = HDF5DataSet(
            hdf5_path=output_paths,
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
            metrics_exporters=[OutputExporter(metrics_directory)],
            transform_sigmoid=True,
        )   

        nn.train(nepoch=10, validate=True) 

        nn.save_model("test.pth.tar")

        NeuralNet(dataset, GINet, pretrained_model="test.pth.tar")

        assert len(os.listdir(metrics_directory)) > 0

    finally:
        rmtree(output_directory)