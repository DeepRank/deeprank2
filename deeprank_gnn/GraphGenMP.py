import os
import logging
import traceback
import sys
import glob
import h5py
from tqdm import tqdm
import time
import multiprocessing as mp
from functools import partial
import pickle

from .preprocess import PreProcessor
from .models.graph import Graph
from .models.query import ProteinProteinInterfaceResidueQuery
from .tools.graph import graph_to_hdf5
from .tools.score import get_all_scores


_log = logging.getLogger(__name__)


class GraphHDF5(object):

    def __init__(self, pdb_path, ref_path=None, graph_type='residue', pssm_path=None,
                 select=None, outfile='graph.hdf5', nproc=1, use_tqdm=True, tmpdir='./',
                 limit=None, biopython=False):
        """Master class from which graphs are computed
        Args:
            pdb_path (str): path to the docking models
            ref_path (str, optional): path to the reference model. Defaults to None.
            graph_type (str, optional): Defaults to 'residue'.
            pssm_path ([type], optional): path to the pssm file. Defaults to None.
            select (str, optional): filter files that starts with 'input'. Defaults to None.
            outfile (str, optional): Defaults to 'graph.hdf5'.
            nproc (int, optional): number of processors. Default to 1.
            use_tqdm (bool, optional): Default to True.
            tmpdir (str, optional): Default to `./`.
            limit (int, optional): Default to None.


            >>> pdb_path = './data/pdb/1ATN/'
            >>> pssm_path = './data/pssm/1ATN/'
            >>> ref = './data/ref/1ATN/'

            >>> GraphHDF5(pdb_path=pdb_path, ref_path=ref, pssm_path=pssm_path,
                          graph_type='residue', outfile='1AK4_residue.hdf5')
        """
        # get the list of PDB names
        pdbs = list(filter(lambda x: x.endswith(
            '.pdb'), os.listdir(pdb_path)))
        if select is not None:
            pdbs = list(filter(lambda x: x.startswith(select), pdbs))

        # get the full path of the pdbs
        pdbs = [os.path.join(pdb_path, name) for name in pdbs]
        if limit is not None:
            if isinstance(limit, list):
                pdbs = pdbs[limit[0]:limit[1]]
            else:
                pdbs = pdbs[:limit]

        # get the pssm data
        pssm_paths = None
        for p in pdbs:
            base = os.path.basename(p)
            mol_name = os.path.splitext(base)[0]
            base_name = mol_name.split('_')[0]
            if pssm_path is not None:
                pssm_paths = self._get_pssm_paths(pssm_path, base_name)

        # get the ref path
        if ref_path is None:
            ref = None
        else:
            ref = os.path.join(ref_path, base_name + '.pdb')

        # compute all the graphs on 1 core and directly
        # store the graphs the HDF5 file
        if nproc == 1:
            graphs = self.get_all_graphs(
                pdbs, pssm_paths, ref, outfile, use_tqdm, biopython)
        else:
            self.preprocess_async(nproc, outfile, pdbs, ref, pssm_paths, biopython)

        # clean up
        rmfiles = glob.glob(
            '*.izone') + glob.glob('*.lzone') + glob.glob('*.refpairs')
        for f in rmfiles:
            os.remove(f)

    def get_all_graphs(self, pdbs, pssm_paths, ref, outfile, use_tqdm=True, biopython=False):

        graphs = []
        if use_tqdm:
            desc = '{:25s}'.format('   Create HDF5')
            lst = tqdm(pdbs, desc=desc, file=sys.stdout)
        else:
            lst = pdbs

        for pdb_path in lst:
            try:
                graphs.append(self._get_one_graph(
                    pdb_path, pssm_paths, ref, biopython))
            except Exception as e:
                print('Issue encountered while computing graph ', pdb_path)
                traceback.print_exc()

        with h5py.File(outfile, 'w') as f5:
            for g in graphs:
                try:
                    graph_to_hdf5(g, f5)
                except Exception as e:
                    print('Issue encountered while storing graph ', g.id)
                    traceback.print_exc()

    @staticmethod
    def preprocess_async(nproc, outfile, pdb_paths, ref_path, pssm_paths, biopython):

        prefix = os.path.splitext(outfile)[0] + "-"
        preprocessor = PreProcessor(prefix, nproc)
        preprocessor.start()
        for pdb_path in pdb_paths:
            targets = {}
            if ref_path is not None:
                targets = get_all_scores(pdb_path, ref_path)

            q = ProteinProteinInterfaceResidueQuery(pdb_path, "A", "B", pssm_paths=pssm_paths,
                                                    targets=targets, use_biopython=biopython)
            preprocessor.add_query(q)
        preprocessor.wait()

        GraphHDF5._combine_hdf5_files_with_prefix(prefix, outfile)

    @staticmethod
    def _combine_hdf5_files_with_prefix(prefix, output_path):

        for input_path in glob.glob(f"{prefix}*.hdf5"):
            with h5py.File(input_path, 'r') as input_file:
                with h5py.File(output_path, 'a') as output_file:
                    for key in input_file:
                        GraphHDF5._copy_hdf5(input_file, key, output_file)

            os.remove(input_path)

    @staticmethod
    def _copy_hdf5(input_, key, output_):

        if type(input_[key]) == h5py.Group:

            out_group = output_.require_group(key)

            for child_key in input_[key]:
                GraphHDF5._copy_hdf5(input_[key], child_key, out_group)

            for key, value in input_[key].attrs.items():
                out_group.attrs[key] = value

        elif type(input_[key]) == h5py.Dataset:

            output_.create_dataset(key, data=input_[key][()])
        else:
            raise TypeError(type(input_[key]))

    @staticmethod
    def _get_one_graph(pdb_path, pssm_paths, ref, biopython):

        targets = {}
        if ref is not None:
            targets = get_all_scores(pdb_path, ref)

        # get the graph

        q = ProteinProteinInterfaceResidueQuery(pdb_path, "A", "B", pssm_paths=pssm_paths,
                                                targets=targets, use_biopython=biopython)

        g = q.build_graph()

        return g

    def _get_pssm_paths(self, pssm_path, pdb_ac):
        return {"A": os.path.join(pssm_path, "{}.A.pdb.pssm".format(pdb_ac)),
                "B": os.path.join(pssm_path, "{}.B.pdb.pssm".format(pdb_ac))}
