import os
import traceback
import sys
import glob
import h5py
from pdb2sql import StructureSimilarity
from tqdm import tqdm
import time
import multiprocessing as mp
from functools import partial
import pickle

from .models.graph import Graph
from .models.query import ProteinProteinInterfaceResidueQuery
from .models.environment import Environment
from .tools.graph import graph_to_hdf5


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

        # get the base name
        for p in pdbs:
            base = os.path.basename(p)
            mol_name = os.path.splitext(base)[0]
            base_name = mol_name.split('_')[0]

        # get the ref path
        if ref_path is None:
            ref = None
        else:
            ref = os.path.join(ref_path, base_name+'.pdb')

        # compute all the graphs on 1 core and directly
        # store the graphs the HDF5 file
        if nproc == 1:
            graphs = self.get_all_graphs(base_name,
                pdbs, pssm_path, ref, outfile, use_tqdm, biopython)

        else:
            if not os.path.isdir(tmpdir):
                os.mkdir(tmpdir)

            pool = mp.Pool(nproc)
            part_process = partial(
                self._pickle_one_graph, model_id=base_name, pssm_root=pssm_path, ref=ref, tmpdir=tmpdir, biopython=biopython)
            pool.map(part_process, pdbs)

            # get teh graph names
            graph_names = [os.path.join(tmpdir, f)
                           for f in os.listdir(tmpdir)]
            graph_names = list(
                filter(lambda x: x.endswith('.pkl'), graph_names))
            if select is not None:
                graph_names = list(
                    filter(lambda x: x.startswith(tmpdir+select), graph_names))

            # transfer them to the hdf5
            with h5py.File(outfile, 'w') as f5:
                desc = '{:25s}'.format('   Store in HDF5')

                for name in graph_names:
                    f = open(name, 'rb')
                    g = pickle.load(f)
                    try:
                        graph_to_hdf5(g, f5)
                    except Exception as e:
                        print(
                            'Issue encountered while computing graph ', name)
                        print(e)
                    f.close()
                    os.remove(name)

        # clean up
        rmfiles = glob.glob(
            '*.izone') + glob.glob('*.lzone') + glob.glob('*.refpairs')
        for f in rmfiles:
            os.remove(f)

    def get_all_graphs(self, model_id, pdbs, pssm, ref, outfile, use_tqdm=True, biopython=False):

        graphs = []
        if use_tqdm:
            desc = '{:25s}'.format('   Create HDF5')
            lst = tqdm(pdbs, desc=desc, file=sys.stdout)
        else:
            lst = pdbs

        for name in lst:
            try:
                graphs.append(self._get_one_graph(model_id,
                    name, pssm, ref, biopython))
            except Exception as e:
                print('Issue encountered while computing graph ', name)
                traceback.print_exc()

        with h5py.File(outfile, 'w') as f5:
            for g in graphs:
                try:
                    graph_to_hdf5(g, f5)
                except Exception as e:
                    print('Issue encountered while storing graph ', g.id)
                    traceback.print_exc()

    @staticmethod
    def _pickle_one_graph(pdb_path, model_id, pssm_root, ref, tmpdir='./', biopython=False):

        environment = Environment(pssm_root=pssm_root)

        # get the graph
        try:

            targets = {}
            if ref is not None:
                targets = GraphHDF5._get_scores(pdb_path, ref)

            q = ProteinProteinInterfaceResidueQuery(model_id, "A", "B",
                                                    targets=targets, use_biopython=biopython)

            g = q.build_graph(pdb_path=pdb_path, environment=environment)

            # pickle it
            fname = os.path.join(tmpdir, '{}.pkl'.format(model_id))

            f = open(fname, 'wb')
            pickle.dump(g, f)
            f.close()

        except Exception as e:
            print('Issue encountered while storing graph ', pdb_path)
            traceback.print_exc()

    @staticmethod
    def _get_one_graph(model_id, pdb_path, pssm_root, ref, biopython):

        environment = Environment(pssm_root=pssm_root)

        targets = {}
        if ref is not None:
            targets = GraphHDF5._get_scores(pdb_path, ref)

        # get the graph

        q = ProteinProteinInterfaceResidueQuery(model_id, "A", "B",
                                                targets=targets, use_biopython=biopython)

        g = q.build_graph(pdb_path=pdb_path, environment=environment)

        return g

    @staticmethod
    def _get_scores(pdb_path, reference_pdb_path):
        """Assigns scores (lrmsd, irmsd, fnat, dockQ, bin_class, capri_class) to a protein graph

        Args:
            pdb_path (path): path to the scored pdb structure
            reference_pdb_path (path): path to the reference structure required to compute the different score
        """

        ref_name = os.path.splitext(os.path.basename(reference_pdb_path))[0]
        sim = StructureSimilarity(pdb_path, reference_pdb_path)

        scores = {}

        # Input pre-computed zone files
        if os.path.exists(ref_name+'.lzone'):
            scores['lrmsd'] = sim.compute_lrmsd_fast(
                method='svd', lzone=ref_name+'.lzone')
            scores['irmsd'] = sim.compute_irmsd_fast(
                method='svd', izone=ref_name+'.izone')

        # Compute zone files
        else:
            scores['lrmsd'] = sim.compute_lrmsd_fast(
                method='svd')
            scores['irmsd'] = sim.compute_irmsd_fast(
                method='svd')

        scores['fnat'] = sim.compute_fnat_fast()
        scores['dockQ'] = sim.compute_DockQScore(
            scores['fnat'], scores['lrmsd'], scores['irmsd'])
        scores['bin_class'] = scores['irmsd'] < 4.0

        scores['capri_class'] = 5
        for thr, val in zip([6.0, 4.0, 2.0, 1.0], [4, 3, 2, 1]):
            if scores['irmsd'] < thr:
                scores['capri_class'] = val

        return scores
