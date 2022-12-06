from deeprankcore.query import QueryCollection, ProteinProteinInterfaceResidueQuery
from multiprocessing import Manager, Pool

queries = QueryCollection()

pdbs = {}
pdbs["1ATN_1w.pdb"] = {}
pdbs["1ATN_1w.pdb"]["pssm_a"] = "1ATN.A.pdb.pssm"
pdbs["1ATN_1w.pdb"]["pssm_b"] = "1ATN.B.pdb.pssm"
pdbs["1ATN_1w.pdb"]["binary"] = 0

pdbs["1ATN_2w.pdb"] = {}
pdbs["1ATN_2w.pdb"]["pssm_a"] = "1ATN.A.pdb.pssm"
pdbs["1ATN_2w.pdb"]["pssm_b"] = "1ATN.B.pdb.pssm"
pdbs["1ATN_2w.pdb"]["binary"] = 0

pdbs["1ATN_3w.pdb"] = {}
pdbs["1ATN_3w.pdb"]["pssm_a"] = "1ATN.A.pdb.pssm"
pdbs["1ATN_3w.pdb"]["pssm_b"] = "1ATN.B.pdb.pssm"
pdbs["1ATN_3w.pdb"]["binary"] = 0

# # not parallelized
# for pdb in pdbs:
#     queries.add(ProteinProteinInterfaceResidueQuery(
#         pdb_path = pdb,
#         chain_id1 = "A",
#         chain_id2 = "B",
#         targets = {
#             "binary": pdbs[pdb]['binary']
#         },
#         pssm_paths = {
#             "A": pdbs[pdb]['pssm_a'],
#             "B": pdbs[pdb]['pssm_b']
#         }
#     ))

# parallelized
def add_query(queries, pdb):
    queries.append(ProteinProteinInterfaceResidueQuery(
        pdb_path = pdb,
        chain_id1 = "A",
        chain_id2 = "B",
        targets = {
            "binary": pdbs[pdb]['binary']
        },
        pssm_paths = {
            "A": pdbs[pdb]['pssm_a'],
            "B": pdbs[pdb]['pssm_b']
        }
    ))

pool = Pool(processes=4)
manager = Manager()
queries = manager.list()

[pool.apply_async(add_query, args=[queries, pdb]) for pdb in pdbs]
print(queries)