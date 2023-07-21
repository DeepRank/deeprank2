# this script allows to copy a small fraction of the real data on a user's folder for fast experimenting with the code
import glob
import shutil
import os
import logging
import sys

####### please modify here #######
models_folder_name = 'HLA_quantitative'
data = 'pMHCI'
src_dir = f'/home/gayatrir/DATA/pdb'
dest_dir = f"/home/ccrocion/repositories/deeprank-core/tutorials/data_raw/variant"
n_files = 1000
##################################

# Loggers
_log = logging.getLogger('')
_log.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
_log.addHandler(sh)

pdb_files = glob.glob(os.path.join(src_dir + '/pdb', '*.pdb'))
count = 0
for pdb_file in pdb_files:
    pdb_id = pdb_file.split('/')[-1].split('.')[0]
    pssm_files = glob.glob(os.path.join(src_dir + '/pssm', pdb_id + '.*.pssm'))
    # make sure to select complete data points
    if len(pssm_files) == 2:
        shutil.copy(pdb_file, dest_dir + '/pdb')
        _log.info(f'copied {pdb_file}')
        for pssm_file in pssm_files:
            shutil.copy(pssm_file, dest_dir + '/pssm')
            _log.info(f'copied {pssm_file}')
        _log.info('\n')
        count += 1
        if count == n_files:
            break
