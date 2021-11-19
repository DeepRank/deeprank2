from pdb2sql import interface


def get_residue_contact_pairs(pdb_path, chain_id1, chain_id2, distance_cutoff):

    pdb_interface = interface(pdb_path)

    for in pdb_interface.get_contact_residues(cutoff=distance_cutoff, chain1=chain_id1, chain2=chain_id2,
                                              return_contact_pairs=True):

        
