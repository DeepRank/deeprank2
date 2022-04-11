import os
import traceback
import sys
import glob
import h5py
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import pickle
from .models.query import ProteinProteinInterfaceResidueQuery
from .tools.graph import graph_to_hdf5
from .tools.score import get_all_scores


class GraphHDF5():
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    def __init__(
        self,
        pdb_path,
        ref_path=None,
        pssm_path=None,
        select=None,
        outfile="graph.hdf5",
        nproc=1,
        tmpdir="./",
        limit=None,
        biopython=False,
    ):
        """Master class from which graphs are computed
        Args:
            pdb_path (str): path to the docking models
            ref_path (str, optional): path to the reference model. Defaults to None.
            pssm_path ([type], optional): path to the pssm file. Defaults to None.
            select (str, optional): filter files that starts with 'input'. Defaults to None.
            outfile (str, optional): Defaults to 'graph.hdf5'.
            nproc (int, optional): number of processors. Default to 1.
            tmpdir (str, optional): Default to `./`.
            limit (int, optional): Default to None.


            >>> pdb_path = './data/pdb/1ATN/'
            >>> pssm_path = './data/pssm/1ATN/'
            >>> ref = './data/ref/1ATN/'

            >>> GraphHDF5(pdb_path=pdb_path, ref_path=ref, pssm_path=pssm_path, outfile='1AK4_residue.hdf5')
        """
        # get the list of PDB names
        pdbs = list(filter(lambda x: x.endswith(".pdb"), os.listdir(pdb_path)))
        if select is not None:
            pdbs = list(filter(lambda x: x.startswith(select), pdbs))

        # get the full path of the pdbs
        pdbs = [os.path.join(pdb_path, name) for name in pdbs]
        if limit is not None:
            if isinstance(limit, list):
                pdbs = pdbs[limit[0] : limit[1]]
            else:
                pdbs = pdbs[:limit]

        # get the pssm data
        pssm_paths = None
        for p in pdbs:
            base = os.path.basename(p)
            mol_name = os.path.splitext(base)[0]
            base_name = mol_name.split("_")[0]
            if pssm_path is not None:
                pssm_paths = self._get_pssm_paths(pssm_path, base_name)

        # get the ref path
        if ref_path is None:
            ref = None
        else:
            ref = os.path.join(ref_path, base_name + ".pdb")

        # compute all the graphs on 1 core and directly
        # store the graphs the HDF5 file

        if nproc != 1:
            if not os.path.isdir(tmpdir):
                os.mkdir(tmpdir)

            pool = mp.Pool(nproc) # pylint: disable=consider-using-with
            part_process = partial(
                self._pickle_one_graph,
                pssm_paths=pssm_paths,
                ref=ref,
                tmpdir=tmpdir,
                biopython=biopython,
            )
            pool.map(part_process, pdbs)

            # get the graph names
            graph_names = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)]
            graph_names = list(filter(lambda x: x.endswith(".pkl"), graph_names))
            if select is not None:
                graph_names = list(
                    filter(lambda x: x.startswith(tmpdir + select), graph_names)
                )

            # transfer them to the hdf5
            with h5py.File(outfile, "w") as f5:

                for name in graph_names:
                    with open(name, "rb") as f:
                        g = pickle.load(f)
                        try:
                            graph_to_hdf5(g, f5)
                        except Exception as e:
                            print("Issue encountered while computing graph ", name)
                            print(e)
                    os.remove(name)

        # clean up
        rmfiles = glob.glob("*.izone") + glob.glob("*.lzone") + glob.glob("*.refpairs")
        for f in rmfiles:
            os.remove(f)

    def get_all_graphs(
        self, pdbs, pssm_paths, ref, outfile, use_tqdm=True, biopython=False
    ):

        graphs = []
        if use_tqdm:
            desc = f"{'   Create HDF5':25s}"
            lst = tqdm(pdbs, desc=desc, file=sys.stdout)
        else:
            lst = pdbs

        for pdb_path in lst:
            try:
                graphs.append(self._get_one_graph(pdb_path, pssm_paths, ref, biopython))
            except Exception:
                print("Issue encountered while computing graph ", pdb_path)
                traceback.print_exc()

        with h5py.File(outfile, "w") as f5:
            for g in graphs:
                try:
                    graph_to_hdf5(g, f5)
                except Exception:
                    print("Issue encountered while storing graph ", g.id)
                    traceback.print_exc()

    @staticmethod
    def _pickle_one_graph(pdb_path, pssm_paths, ref, tmpdir="./", biopython=False):

        # get the graph
        try:

            targets = {}
            if ref is not None:
                targets = get_all_scores(pdb_path, ref)

            q = ProteinProteinInterfaceResidueQuery(
                pdb_path,
                "A",
                "B",
                pssm_paths=pssm_paths,
                targets=targets,
                use_biopython=biopython,
            )

            g = q.build_graph()

            # pickle it
            fname = os.path.join(tmpdir, f"{g.id}.pkl")

            with open(fname, "wb") as f:
                pickle.dump(g, f)

        except Exception:
            print("Issue encountered while storing graph ", pdb_path)
            traceback.print_exc()

    @staticmethod
    def _get_one_graph(pdb_path, pssm_paths, ref, biopython):

        targets = {}
        if ref is not None:
            targets = get_all_scores(pdb_path, ref)

        # get the graph

        q = ProteinProteinInterfaceResidueQuery(
            pdb_path,
            "A",
            "B",
            pssm_paths=pssm_paths,
            targets=targets,
            use_biopython=biopython,
        )

        g = q.build_graph()

        return g

    def _get_pssm_paths(self, pssm_path, pdb_ac):
        return {
            "A": os.path.join(pssm_path, f"{pdb_ac}.A.pdb.pssm"),
            "B": os.path.join(pssm_path, f"{pdb_ac}.B.pdb.pssm"),
        }
