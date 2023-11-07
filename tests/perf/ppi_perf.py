# This script can be used for performance testing of the DeepRank2 package, using the PPI query classes.
import glob
import os
import time
from os import listdir
from os.path import isfile, join

import numpy
import pandas as pd

from deeprank2.features import components, contact, exposure, irc, secondary_structure, surfacearea
from deeprank2.query import ProteinProteinInterfaceAtomicQuery, QueryCollection
from deeprank2.utils.grid import GridSettings, MapMethod

#################### PARAMETERS ####################
interface_distance_cutoff = 5.5  # max distance in Å between two interacting residues/atoms of two proteins
grid_settings = GridSettings(  # None if you don't want grids
    # the number of points on the x, y, z edges of the cube
    points_counts=[35, 30, 30],
    # x, y, z sizes of the box in Å
    sizes=[1.0, 1.0, 1.0],
)
grid_map_method = MapMethod.GAUSSIAN  # None if you don't want grids
# grid_settings = None
# grid_map_method = None
feature_modules = [components, contact, exposure, irc, secondary_structure, surfacearea]
cpu_count = 1
####################################################

data_path = os.path.join("data_raw", "ppi")
processed_data_path = os.path.join("data_processed", "ppi")

if not os.path.exists(os.path.join(processed_data_path, "atomic")):
    os.makedirs(os.path.join(processed_data_path, "atomic"))


def get_pdb_files_and_target_data(data_path):
    csv_data = pd.read_csv(os.path.join(data_path, "BA_values.csv"))
    pdb_files = glob.glob(os.path.join(data_path, "pdb", "*.pdb"))
    pdb_files.sort()
    pdb_ids_csv = [pdb_file.split("/")[-1].split(".")[0] for pdb_file in pdb_files]
    csv_data_indexed = csv_data.set_index("ID")
    csv_data_indexed = csv_data_indexed.loc[pdb_ids_csv]
    bas = csv_data_indexed.measurement_value.values.tolist()
    return pdb_files, bas


if __name__ == "__main__":
    timings = []
    count = 0
    pdb_files, bas = get_pdb_files_and_target_data(data_path)

    for i, pdb_file in enumerate(pdb_files):
        queries = QueryCollection()
        queries.add(
            ProteinProteinInterfaceAtomicQuery(
                pdb_path=pdb_file,
                chain_id1="M",
                chain_id2="P",
                distance_cutoff=interface_distance_cutoff,
                targets={
                    "binary": int(float(bas[i]) <= 500),  # binary target value
                    "BA": bas[i],  # continuous target value
                },
            )
        )

        start = time.perf_counter()
        queries.process(
            prefix=os.path.join(processed_data_path, "atomic", "proc"),
            feature_modules=feature_modules,
            cpu_count=cpu_count,
            combine_output=False,
            grid_settings=grid_settings,
            grid_map_method=grid_map_method,
        )
        end = time.perf_counter()
        elapsed = end - start
        timings.append(elapsed)
        print(f"Elapsed time: {elapsed:.6f} seconds.\n")

    timings = numpy.array(timings)
    print(f'The queries processing is done. The generated HDF5 files are in {os.path.join(processed_data_path, "atomic")}.')
    print(f"Avg: {numpy.mean(timings):.6f} seconds.")
    print(f"Std: {numpy.std(timings):.6f} seconds.\n")

    proc_files_path = os.path.join(processed_data_path, "atomic")
    proc_files = [f for f in listdir(proc_files_path) if isfile(join(proc_files_path, f))]
    mem_sizes = []
    for proc_file in proc_files:
        file_size = os.path.getsize(os.path.join(proc_files_path, proc_file))
        mb_file_size = file_size / (10**6)
        print(f"Size of {proc_file}: {mb_file_size} MB.\n")
        mem_sizes.append(mb_file_size)
    mem_sizes = numpy.array(mem_sizes)
    print(f"Avg: {numpy.mean(mem_sizes):.6f} MB.")
    print(f"Std: {numpy.std(mem_sizes):.6f} MB.")
