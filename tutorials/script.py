import csv

with open('sample', 'r') as file:
    reader = csv.reader(file)
    #next(reader)  # Skip header line
    
    for row in reader:
        col1, col2, col3, col4, col5 = row
        
        #variant aa
        col2_first_three = col2[:3]
        col2_last_three = col2[-3:]
        
        #pdb, chain
        col4_first_four = col4[:4]
        col4_last = col4[-1]
        
        print(f'queries.add(SingleResidueVariantAtomicQuery(')
        print(f'    pdb_path = "/home/gayatrir/DATA/pdb/pdb{col4_first_four}.ent",')
        print(f'    chain_id = "{col4_last}",')
        print(f'    residue_number={col3},')
        print(f'    insertion_code = None,')
        print(f'    wildtype_amino_acid = "{col2_first_three}",')
        print(f'    variant_amino_acid = "{col2_last_three}",')
        print(f'    pssm_paths = {{"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/{col4_first_four}.{col4_last}.pdb.pssm"}},')
        print(f'    targets={{targets.BINARY: {col5}}},')
        print(f'    radius= 10.0,')
        print(f'    distance_cutoff= 4.5,')
        print(f'))')
        print(f'    ')

