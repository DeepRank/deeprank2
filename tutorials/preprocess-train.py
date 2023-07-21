import sys
import os
import logging

import deeprankcore
from deeprankcore.query import QueryCollection, SingleResidueVariantAtomicQuery
from deeprankcore.features import components, conservation, contact, surfacearea
from deeprankcore.utils.grid import GridSettings, MapMethod
from deeprankcore.domain.aminoacidlist import (alanine, arginine, asparagine,
                                               serine, glycine, leucine, aspartate,
                                               glutamine, glutamate, lysine, phenylalanine, histidine,
                                               tyrosine, tryptophan, valine, proline,
                                               cysteine, isoleucine, methionine, threonine)
from deeprankcore.domain import targetstorage as targets


queries = QueryCollection()

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zw9.ent",
    chain_id = "A",
    residue_number = 50,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = alanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zw9.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zw9.ent",
    chain_id = "A",
    residue_number = 54,
    insertion_code = None,
    wildtype_amino_acid = proline,
    variant_amino_acid = serine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zw9.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zw9.ent",
    chain_id = "A",
    residue_number = 91,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = histidine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zw9.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zw9.ent",
    chain_id = "A",
    residue_number = 227,
    insertion_code = None,
    wildtype_amino_acid = glutamine,
    variant_amino_acid = leucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zw9.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4fjv.ent",
    chain_id = "A",
    residue_number = 6,
    insertion_code = None,
    wildtype_amino_acid = phenylalanine,
    variant_amino_acid = cysteine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4fjv.A.pdb.pssm",
        "B" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4fjv.B.pdb.pssm",
        "C" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4fjv.C.pdb.pssm",
        "D" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4fjv.D.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4fjv.ent",
    chain_id = "A",
    residue_number = 20,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = tryptophan,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4fjv.A.pdb.pssm",
        "B" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4fjv.B.pdb.pssm",
        "C" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4fjv.C.pdb.pssm",
        "D" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4fjv.D.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3lxb.ent",
    chain_id = "A",
    residue_number = 34,
    insertion_code = None,
    wildtype_amino_acid = asparagine,
    variant_amino_acid = serine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3lxb.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number = 37,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = proline,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm",
        "B" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.B.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number = 40,
    insertion_code = None,
    wildtype_amino_acid = proline,
    variant_amino_acid = leucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm",
        "B" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.B.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb2g98.ent",
    chain_id = "A",
    residue_number = 14,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = cysteine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/2g98.A.pdb.pssm",
        "B": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/2g98.B.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1bzy.ent",
    chain_id = "A",
    residue_number = 47,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = histidine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1bzy.A.pdb.pssm",
        "B" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1bzy.B.pdb.pssm",
        "C" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1bzy.C.pdb.pssm",
        "D" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1bzy.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1bzy.ent",
    chain_id = "A",
    residue_number= 49,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1bzy.A.pdb.pssm",
        "B" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1bzy.B.pdb.pssm",
        "C" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1bzy.C.pdb.pssm",
        "D" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1bzy.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1e2d.ent",
    chain_id = "A",
    residue_number= 165,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1e2d.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1n3u.ent",
    chain_id = "A",
    residue_number= 193,
    insertion_code = None,
    wildtype_amino_acid = proline,
    variant_amino_acid = serine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1n3u.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1gzw.ent",
    chain_id = "A",
    residue_number= 71,
    insertion_code = None,
    wildtype_amino_acid = glutamate,
    variant_amino_acid = lysine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1gzw.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4hwk.ent",
    chain_id = "A",
    residue_number= 147,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = glycine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4hwk.A.pdb.pssm",
        "B" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4hwk.B.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4hwk.ent",
    chain_id = "A",
    residue_number= 172,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = aspartate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4hwk.A.pdb.pssm",
        "B" : "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4hwk.B.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3uzd.ent",
    chain_id = "A",
    residue_number= 129,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = glutamate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3uzd.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3uzd.ent",
    chain_id = "A",
    residue_number= 99,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = asparagine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3uzd.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3uzd.ent",
    chain_id = "A",
    residue_number= 132,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = histidine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3uzd.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5c65.ent",
    chain_id = "A",
    residue_number= 91,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = histidine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5c65.A.pdb.pssm",
        "B": "/mnt/csb/DeepRank-Mut-DATA/pssm/update/5c65.B.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1tff.ent",
    chain_id = "A",
    residue_number= 6,
    insertion_code = None,
    wildtype_amino_acid = phenylalanine,
    variant_amino_acid = cysteine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1tff.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1tff.ent",
    chain_id = "A",
    residue_number= 20,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = tryptophan,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1tff.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1tff.ent",
    chain_id = "A",
    residue_number= 78,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = isoleucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1tff.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1ca7.ent",
    chain_id = "A",
    residue_number= 112,
    insertion_code = None,
    wildtype_amino_acid = threonine,
    variant_amino_acid = isoleucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1ca7.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1gcz.ent",
    chain_id = "A",
    residue_number= 112,
    insertion_code = None,
    wildtype_amino_acid = threonine,
    variant_amino_acid = isoleucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1gcz.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3smb.ent",
    chain_id = "A",
    residue_number= 112,
    insertion_code = None,
    wildtype_amino_acid = threonine,
    variant_amino_acid = isoleucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3smb.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4gum.ent",
    chain_id = "A",
    residue_number= 112,
    insertion_code = None,
    wildtype_amino_acid = threonine,
    variant_amino_acid = isoleucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4gum.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5bs9.ent",
    chain_id = "A",
    residue_number= 112,
    insertion_code = None,
    wildtype_amino_acid = threonine,
    variant_amino_acid = isoleucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5bs9.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb6b1k.ent",
    chain_id = "A",
    residue_number= 112,
    insertion_code = None,
    wildtype_amino_acid = threonine,
    variant_amino_acid = isoleucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/6b1k.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb6peg.ent",
    chain_id = "A",
    residue_number= 112,
    insertion_code = None,
    wildtype_amino_acid = threonine,
    variant_amino_acid = isoleucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/6peg.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1gif.ent",
    chain_id = "A",
    residue_number= 112,
    insertion_code = None,
    wildtype_amino_acid = threonine,
    variant_amino_acid = isoleucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1gif.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1gzw.ent",
    chain_id = "A",
    residue_number= 71,
    insertion_code = None,
    wildtype_amino_acid = glutamate,
    variant_amino_acid = lysine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1gzw.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3oy8.ent",
    chain_id = "A",
    residue_number= 71,
    insertion_code = None,
    wildtype_amino_acid = glutamate,
    variant_amino_acid = lysine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3oy8.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4y24.ent",
    chain_id = "A",
    residue_number= 71,
    insertion_code = None,
    wildtype_amino_acid = glutamate,
    variant_amino_acid = lysine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4y24.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3t2t.ent",
    chain_id = "A",
    residue_number= 72,
    insertion_code = None,
    wildtype_amino_acid = glutamate,
    variant_amino_acid = lysine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3t2t.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1n3u.ent",
    chain_id = "A",
    residue_number=29,
    insertion_code = None,
    wildtype_amino_acid = glutamate,
    variant_amino_acid = aspartate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1n3u.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1oyk.ent",
    chain_id = "A",
    residue_number=29,
    insertion_code = None,
    wildtype_amino_acid = glutamate,
    variant_amino_acid = aspartate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1oyk.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1ozr.ent",
    chain_id = "A",
    residue_number=29,
    insertion_code = None,
    wildtype_amino_acid = glutamate,
    variant_amino_acid = aspartate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1ozr.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1n3u.ent",
    chain_id = "A",
    residue_number=193,
    insertion_code = None,
    wildtype_amino_acid = proline,
    variant_amino_acid = serine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1n3u.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3czy.ent",
    chain_id = "A",
    residue_number=193,
    insertion_code = None,
    wildtype_amino_acid = proline,
    variant_amino_acid = serine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3czy.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number=40,
    insertion_code = None,
    wildtype_amino_acid = proline,
    variant_amino_acid = leucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3hg5.ent",
    chain_id = "A",
    residue_number=40,
    insertion_code = None,
    wildtype_amino_acid = proline,
    variant_amino_acid = leucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3hg5.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb6ibt.ent",
    chain_id = "A",
    residue_number=40,
    insertion_code = None,
    wildtype_amino_acid = proline,
    variant_amino_acid = serine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/6ibt.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number=42,
    insertion_code = None,
    wildtype_amino_acid = methionine,
    variant_amino_acid = threonine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3tv8.ent",
    chain_id = "A",
    residue_number=42,
    insertion_code = None,
    wildtype_amino_acid = methionine,
    variant_amino_acid = threonine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3tv8.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number=42,
    insertion_code = None,
    wildtype_amino_acid = methionine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number=43,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3hg5.ent",
    chain_id = "A",
    residue_number=43,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3hg5.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number=46,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = arginine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3hg5.ent",
    chain_id = "A",
    residue_number=46,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = arginine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3hg5.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number=89,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = arginine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3hg5.ent",
    chain_id = "A",
    residue_number=52,
    insertion_code = None,
    wildtype_amino_acid = cysteine,
    variant_amino_acid = glycine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3hg5.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number=52,
    insertion_code = None,
    wildtype_amino_acid = cysteine,
    variant_amino_acid = arginine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3tv8.ent",
    chain_id = "A",
    residue_number=51,
    insertion_code = None,
    wildtype_amino_acid = methionine,
    variant_amino_acid = isoleucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3tv8.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3hg5.ent",
    chain_id = "A",
    residue_number=47,
    insertion_code = None,
    wildtype_amino_acid = tryptophan,
    variant_amino_acid = cysteine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3hg5.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
    ))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number=106,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = arginine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number=223,
    insertion_code = None,
    wildtype_amino_acid = cysteine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number=205,
    insertion_code = None,
    wildtype_amino_acid = proline,
    variant_amino_acid = leucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3hg5.ent",
    chain_id = "A",
    residue_number=147,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = arginine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3hg5.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb2uuh.ent",
    chain_id = "A",
    residue_number=125,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = aspartate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/2uuh.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb2uuh.ent",
    chain_id = "A",
    residue_number=26,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = leucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/2uuh.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1t7v.ent",
    chain_id = "A",
    residue_number=198,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = glutamate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1t7v.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1t7v.ent",
    chain_id = "A",
    residue_number=183,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = glutamine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1t7v.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1e2d.ent",
    chain_id = "A",
    residue_number=165,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1e2d.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4ktt.ent",
    chain_id = "A",
    residue_number=161,
    insertion_code = None,
    wildtype_amino_acid = asparagine,
    variant_amino_acid = lysine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4ktt.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4ktt.ent",
    chain_id = "A",
    residue_number=216,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = glycine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4ktt.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb2nup.ent",
    chain_id = "A",
    residue_number=167,
    insertion_code = None,
    wildtype_amino_acid = isoleucine,
    variant_amino_acid = threonine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/2nup.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb2nup.ent",
    chain_id = "A",
    residue_number=408,
    insertion_code = None,
    wildtype_amino_acid = isoleucine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/2nup.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3efo.ent",
    chain_id = "A",
    residue_number=636,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3efo.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=15,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=47,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = histidine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=48,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=49,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = proline,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=50,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = glutamine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=63,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = aspartate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=67,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=69,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = glutamate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=70,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = arginine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=73,
    insertion_code = None,
    wildtype_amino_acid = phenylalanine,
    variant_amino_acid = cysteine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=160,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = threonine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=166,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = threonine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=188,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = alanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=193,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = histidine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1hmp.ent",
    chain_id = "A",
    residue_number=200,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1hmp.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=15,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=47,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = histidine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=48,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=49,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = proline,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=50,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = glutamine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 0},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=63,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = aspartate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=67,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=69,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = glutamate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=70,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = arginine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=73,
    insertion_code = None,
    wildtype_amino_acid = phenylalanine,
    variant_amino_acid = cysteine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=160,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = threonine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=166,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = threonine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=188,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = alanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=193,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = histidine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))
    
queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1z7g.ent",
    chain_id = "A",
    residue_number=200,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1z7g.A.pdb.pssm"},
    targets={targets.BINARY: 1},
    radius= 10.0,
    distance_cutoff= 4.5,
))



feature_modules = [components, conservation, contact, surfacearea]

hdf5_paths = queries.process(
    "/home/gayatrir/core-train",
    feature_modules = feature_modules,
    cpu_count= 10,
    combine_output= True,
    grid_settings = GridSettings(
        points_counts = [20, 20, 20],
        sizes = [1.0, 1.0, 1.0]),
    grid_map_method = MapMethod.GAUSSIAN,
    grid_augmentation_count= 10)


