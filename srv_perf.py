import glob
import os
import time
from os import listdir
from os.path import isfile, join

import numpy
import pandas as pd

from deeprank2.domain.aminoacidlist import (alanine, arginine, asparagine,
                                            aspartate, cysteine, glutamate,
                                            glutamine, glycine, histidine,
                                            isoleucine, leucine, lysine,
                                            methionine, phenylalanine, proline,
                                            serine, threonine, tryptophan,
                                            tyrosine, valine)
from deeprank2.features import components, contact, exposure, irc, surfacearea
from deeprank2.query import QueryCollection, SingleResidueVariantResidueQuery
from deeprank2.utils.grid import GridSettings, MapMethod

aa_dict = {"ALA": alanine, "CYS": cysteine, "ASP": aspartate,
           "GLU": glutamate, "PHE": phenylalanine, "GLY": glycine,
           "HIS": histidine, "ILE": isoleucine, "LYS": lysine,
           "LEU": leucine, "MET": methionine, "ASN": asparagine,
           "PRO": proline, "GLN": glutamine, "ARG": arginine,
           "SER": serine, "THR": threonine, "VAL": valine,
           "TRP": tryptophan, "TYR": tyrosine
           }

#################### PARAMETERS ####################
radius = 10.0
distance_cutoff = 5.5
grid_settings = GridSettings( # None if you don't want grids
	# the number of points on the x, y, z edges of the cube
	points_counts = [35, 30, 30],
	# x, y, z sizes of the box in Å
	sizes = [1.0, 1.0, 1.0])
grid_map_method = MapMethod.GAUSSIAN # None if you don't want grids
# grid_settings = None
# grid_map_method = None
feature_modules = [components, contact, exposure, irc, surfacearea]
cpu_count = 1
####################################################

data_path = os.path.join("data_raw", "srv")
processed_data_path = os.path.join("data_processed", "srv")

if not os.path.exists(os.path.join(processed_data_path, "atomic")):
	os.makedirs(os.path.join(processed_data_path, "atomic"))

def get_pdb_files_and_target_data(data_path):
	csv_data = pd.read_csv(os.path.join(data_path, "srv_target_values.csv"))
	pdb_files = glob.glob(os.path.join(data_path, "pdb", '*.ent'))
	pdb_files.sort()
	pdb_file_names = [os.path.basename(pdb_file) for pdb_file in pdb_files]
	csv_data_indexed = csv_data.set_index('pdb_file')
	csv_data_indexed = csv_data_indexed.loc[pdb_file_names]
	res_numbers = csv_data_indexed.res_number.values.tolist()
	res_wildtypes = csv_data_indexed.res_wildtype.values.tolist()
	res_variants = csv_data_indexed.res_variant.values.tolist()
	targets = csv_data_indexed.target.values.tolist()
	pdb_names = csv_data_indexed.index.values.tolist()
	pdb_files = [data_path + "/pdb/" + pdb_name for pdb_name in pdb_names]
	return pdb_files, res_numbers, res_wildtypes, res_variants, targets


if __name__=='__main__':

	timings = []
	count = 0
	pdb_files, res_numbers, res_wildtypes, res_variants, targets = get_pdb_files_and_target_data(data_path)
	pdb_files = pdb_files[:10]

	for i in range(len(pdb_files)):
		queries = QueryCollection()
		queries.add(
			SingleResidueVariantResidueQuery(
				pdb_path = pdb_files[i],
				chain_id = "A",
				residue_number = res_numbers[i],
				insertion_code = None,
				wildtype_amino_acid = aa_dict[res_wildtypes[i]],
				variant_amino_acid = aa_dict[res_variants[i]],
				targets = {'binary': targets[i]},
				radius = radius,
				distance_cutoff = distance_cutoff,
			))

		start = time.perf_counter()
		queries.process(
			prefix = os.path.join(processed_data_path, "atomic", "proc"),
			feature_modules = feature_modules,
			cpu_count = cpu_count,
			combine_output = False,
			grid_settings = grid_settings,
			grid_map_method = grid_map_method)
		end = time.perf_counter()
		elapsed = end - start
		timings.append(elapsed)
		print(f'Elapsed time: {elapsed:.6f} seconds.\n')

	timings = numpy.array(timings)
	print(f'The queries processing is done. The generated HDF5 files are in {os.path.join(processed_data_path, "atomic")}.')
	print(f'Avg: {numpy.mean(timings):.6f} seconds.')
	print(f'Std: {numpy.std(timings):.6f} seconds.\n')

	proc_files_path = os.path.join(processed_data_path, "atomic")
	proc_files = [f for f in listdir(proc_files_path) if isfile(join(proc_files_path, f))]
	mem_sizes = []
	for proc_file in proc_files:
		file_size = os.path.getsize(os.path.join(proc_files_path, proc_file))
		mb_file_size = file_size / (10**6)
		print(f'Size of {proc_file}: {mb_file_size} MB.\n')
		mem_sizes.append(mb_file_size)
	mem_sizes = numpy.array(mem_sizes)
	print(f'Avg: {numpy.mean(mem_sizes):.6f} MB.')
	print(f'Std: {numpy.std(mem_sizes):.6f} MB.')
