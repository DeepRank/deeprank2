# This script can be used for performance testing of the DeepRank2 package, using the SRV query classes.
import glob
import os
import time
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

from deeprank2.domain.aminoacidlist import (
    alanine,
    arginine,
    asparagine,
    aspartate,
    cysteine,
    glutamate,
    glutamine,
    glycine,
    histidine,
    isoleucine,
    leucine,
    lysine,
    methionine,
    phenylalanine,
    proline,
    serine,
    threonine,
    tryptophan,
    tyrosine,
    valine,
)
from deeprank2.features import (
    components,
    contact,
    exposure,
    irc,
    secondary_structure,
    surfacearea,
)
from deeprank2.query import QueryCollection, SingleResidueVariantResidueQuery
from deeprank2.utils.grid import GridSettings, MapMethod

aa_dict = {
    "ALA": alanine,
    "CYS": cysteine,
    "ASP": aspartate,
    "GLU": glutamate,
    "PHE": phenylalanine,
    "GLY": glycine,
    "HIS": histidine,
    "ILE": isoleucine,
    "LYS": lysine,
    "LEU": leucine,
    "MET": methionine,
    "ASN": asparagine,
    "PRO": proline,
    "GLN": glutamine,
    "ARG": arginine,
    "SER": serine,
    "THR": threonine,
    "VAL": valine,
    "TRP": tryptophan,
    "TYR": tyrosine,
}

#################### PARAMETERS ####################
radius = 10.0
distance_cutoff = 5.5
grid_settings = GridSettings(  # None if you don't want grids
    # the number of points on the x, y, z edges of the cube
    points_counts=[35, 30, 30],
    # x, y, z sizes of the box in Å
    sizes=[1.0, 1.0, 1.0],
)
grid_map_method = MapMethod.GAUSSIAN  # None if you don't want grids
# grid_settings = None  # noqa: ERA001
# grid_map_method = None  # noqa: ERA001
feature_modules = [components, contact, exposure, irc, surfacearea, secondary_structure]
cpu_count = 1
####################################################

data_path = os.path.join("data_raw", "srv")
processed_data_path = os.path.join("data_processed", "srv")

if not os.path.exists(os.path.join(processed_data_path, "atomic")):
    os.makedirs(os.path.join(processed_data_path, "atomic"))


def get_pdb_files_and_target_data(data_path: str) -> tuple[list[str], list, list, list, list]:
    csv_data = pd.read_csv(os.path.join(data_path, "srv_target_values.csv"))
    # before running this script change .ent to .pdb
    pdb_files = glob.glob(os.path.join(data_path, "pdb", "*.pdb"))
    pdb_files.sort()
    pdb_id = [os.path.basename(pdb_file).split(".")[0] for pdb_file in pdb_files]
    csv_data["pdb_id"] = csv_data["pdb_file"].apply(lambda x: x.split(".")[0])
    csv_data_indexed = csv_data.set_index("pdb_id")
    csv_data_indexed = csv_data_indexed.loc[pdb_id]
    res_numbers = csv_data_indexed.res_number.to_numpy().tolist()
    res_wildtypes = csv_data_indexed.res_wildtype.to_numpy().tolist()
    res_variants = csv_data_indexed.res_variant.to_numpy().tolist()
    targets = csv_data_indexed.target.to_numpy().tolist()
    pdb_names = csv_data_indexed.index.to_numpy().tolist()
    pdb_files = [data_path + "/pdb/" + pdb_name + ".pdb" for pdb_name in pdb_names]
    return pdb_files, res_numbers, res_wildtypes, res_variants, targets


if __name__ == "__main__":
    timings = []
    count = 0
    (
        pdb_files,
        res_numbers,
        res_wildtypes,
        res_variants,
        targets,
    ) = get_pdb_files_and_target_data(data_path)

    for i, pdb_file in enumerate(pdb_files):
        queries = QueryCollection()
        queries.add(
            SingleResidueVariantResidueQuery(
                pdb_path=pdb_file,
                chain_id="A",
                residue_number=res_numbers[i],
                insertion_code=None,
                wildtype_amino_acid=aa_dict[res_wildtypes[i]],
                variant_amino_acid=aa_dict[res_variants[i]],
                targets={"binary": targets[i]},
                radius=radius,
                distance_cutoff=distance_cutoff,
            ),
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

    timings = np.array(timings)
    print(f'The queries processing is done. The generated HDF5 files are in {os.path.join(processed_data_path, "atomic")}.')
    print(f"Avg: {np.mean(timings):.6f} seconds.")
    print(f"Std: {np.std(timings):.6f} seconds.\n")

    proc_files_path = os.path.join(processed_data_path, "atomic")
    proc_files = [f for f in listdir(proc_files_path) if isfile(join(proc_files_path, f))]
    mem_sizes = []
    for proc_file in proc_files:
        file_size = os.path.getsize(os.path.join(proc_files_path, proc_file))
        mb_file_size = file_size / (10**6)
        print(f"Size of {proc_file}: {mb_file_size} MB.\n")
        mem_sizes.append(mb_file_size)
    mem_sizes = np.array(mem_sizes)
    print(f"Avg: {np.mean(mem_sizes):.6f} MB.")
    print(f"Std: {np.std(mem_sizes):.6f} MB.")
