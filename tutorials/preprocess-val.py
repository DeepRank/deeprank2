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
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5ltw.ent",
    chain_id = "C",
    residue_number=104,
    insertion_code = None,
    wildtype_amino_acid = glutamate,
    variant_amino_acid = lysine,
    pssm_paths = {"C": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5ltw.C.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5ltw.ent",
    chain_id = "C",
    residue_number=10,
    insertion_code = None,
    wildtype_amino_acid = serine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"C": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5ltw.C.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5ltw.ent",
    chain_id = "C",
    residue_number=92,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = alanine,
    pssm_paths = {"C": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5ltw.C.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zw9.ent",
    chain_id = "A",
    residue_number=91,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = histidine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zw9.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zwb.ent",
    chain_id = "A",
    residue_number=91,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = histidine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zwb.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zw9.ent",
    chain_id = "A",
    residue_number=244,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = glutamate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zw9.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zwb.ent",
    chain_id = "A",
    residue_number=244,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = glutamate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zwb.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zwc.ent",
    chain_id = "A",
    residue_number=244,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = glutamate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zwc.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5c65.ent",
    chain_id = "A",
    residue_number=244,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = glutamate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5c65.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zwb.ent",
    chain_id = "A",
    residue_number=227,
    insertion_code = None,
    wildtype_amino_acid = glutamine,
    variant_amino_acid = leucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zwb.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zwc.ent",
    chain_id = "A",
    residue_number=227,
    insertion_code = None,
    wildtype_amino_acid = glutamine,
    variant_amino_acid = leucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zwc.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5c65.ent",
    chain_id = "A",
    residue_number=227,
    insertion_code = None,
    wildtype_amino_acid = glutamine,
    variant_amino_acid = leucine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5c65.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zwb.ent",
    chain_id = "A",
    residue_number=50,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = alanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zwb.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zwc.ent",
    chain_id = "A",
    residue_number=50,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = alanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zwc.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5c65.ent",
    chain_id = "A",
    residue_number=50,
    insertion_code = None,
    wildtype_amino_acid = glycine,
    variant_amino_acid = alanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5c65.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zw9.ent",
    chain_id = "A",
    residue_number=54,
    insertion_code = None,
    wildtype_amino_acid = proline,
    variant_amino_acid = serine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zw9.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4zwc.ent",
    chain_id = "A",
    residue_number=54,
    insertion_code = None,
    wildtype_amino_acid = proline,
    variant_amino_acid = serine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4zwc.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5c65.ent",
    chain_id = "A",
    residue_number=54,
    insertion_code = None,
    wildtype_amino_acid = proline,
    variant_amino_acid = serine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5c65.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5qio.ent",
    chain_id = "A",
    residue_number=20,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = tryptophan,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5qio.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5qip.ent",
    chain_id = "A",
    residue_number=20,
    insertion_code = None,
    wildtype_amino_acid = arginine,
    variant_amino_acid = tryptophan,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5qip.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5vnk.ent",
    chain_id = "A",
    residue_number=636,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5vnk.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5vnl.ent",
    chain_id = "A",
    residue_number=636,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5vnl.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5vnm.ent",
    chain_id = "A",
    residue_number=636,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5vnm.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5vnn.ent",
    chain_id = "A",
    residue_number=636,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5vnn.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5vno.ent",
    chain_id = "A",
    residue_number=636,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5vno.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3fvx.ent",
    chain_id = "A",
    residue_number=290,
    insertion_code = None,
    wildtype_amino_acid = glutamate,
    variant_amino_acid = lysine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3fvx.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4wlh.ent",
    chain_id = "A",
    residue_number=290,
    insertion_code = None,
    wildtype_amino_acid = glutamate,
    variant_amino_acid = lysine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4wlh.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4wlj.ent",
    chain_id = "A",
    residue_number=290,
    insertion_code = None,
    wildtype_amino_acid = glutamate,
    variant_amino_acid = lysine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4wlj.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1w7l.ent",
    chain_id = "A",
    residue_number=120,
    insertion_code = None,
    wildtype_amino_acid = isoleucine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1w7l.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1w7m.ent",
    chain_id = "A",
    residue_number=120,
    insertion_code = None,
    wildtype_amino_acid = isoleucine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1w7m.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1w7n.ent",
    chain_id = "A",
    residue_number=120,
    insertion_code = None,
    wildtype_amino_acid = isoleucine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1w7n.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3fvs.ent",
    chain_id = "A",
    residue_number=120,
    insertion_code = None,
    wildtype_amino_acid = isoleucine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3fvs.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3fvu.ent",
    chain_id = "A",
    residue_number=120,
    insertion_code = None,
    wildtype_amino_acid = isoleucine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3fvu.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3fvx.ent",
    chain_id = "A",
    residue_number=120,
    insertion_code = None,
    wildtype_amino_acid = isoleucine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3fvx.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4wlh.ent",
    chain_id = "A",
    residue_number=120,
    insertion_code = None,
    wildtype_amino_acid = isoleucine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4wlh.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4wlj.ent",
    chain_id = "A",
    residue_number=120,
    insertion_code = None,
    wildtype_amino_acid = isoleucine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4wlj.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1w7l.ent",
    chain_id = "A",
    residue_number=358,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = methionine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1w7l.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1w7m.ent",
    chain_id = "A",
    residue_number=358,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = methionine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1w7m.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1w7n.ent",
    chain_id = "A",
    residue_number=358,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = methionine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1w7n.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3fvs.ent",
    chain_id = "A",
    residue_number=358,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = methionine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3fvs.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3fvu.ent",
    chain_id = "A",
    residue_number=358,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = methionine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3fvu.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3fvx.ent",
    chain_id = "A",
    residue_number=358,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = methionine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3fvx.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4wlh.ent",
    chain_id = "A",
    residue_number=358,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = methionine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4wlh.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb4wlj.ent",
    chain_id = "A",
    residue_number=358,
    insertion_code = None,
    wildtype_amino_acid = valine,
    variant_amino_acid = methionine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/4wlj.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1e2e.ent",
    chain_id = "A",
    residue_number=165,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1e2e.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1e2g.ent",
    chain_id = "A",
    residue_number=35,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1e2g.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1e2q.ent",
    chain_id = "A",
    residue_number=35,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1e2q.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1e98.ent",
    chain_id = "A",
    residue_number=35,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1e98.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1e99.ent",
    chain_id = "A",
    residue_number=35,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1e99.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1e9a.ent",
    chain_id = "A",
    residue_number=35,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1e9a.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1e9b.ent",
    chain_id = "A",
    residue_number=35,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1e9b.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1e9c.ent",
    chain_id = "A",
    residue_number=35,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1e9c.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1e9d.ent",
    chain_id = "A",
    residue_number=35,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1e9d.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1e9e.ent",
    chain_id = "A",
    residue_number=35,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1e9e.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1nmx.ent",
    chain_id = "A",
    residue_number=35,
    insertion_code = None,
    wildtype_amino_acid = histidine,
    variant_amino_acid = tyrosine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1nmx.A.pdb.pssm"},
    targets={targets.BINARY: 0.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxp.ent",
    chain_id = "A",
    residue_number=135,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxp.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3gxt.ent",
    chain_id = "A",
    residue_number=135,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3gxt.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3hg2.ent",
    chain_id = "A",
    residue_number=135,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3hg2.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3hg3.ent",
    chain_id = "A",
    residue_number=135,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3hg3.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3hg4.ent",
    chain_id = "A",
    residue_number=135,
    insertion_code = None,
    wildtype_amino_acid = alanine,
    variant_amino_acid = valine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3hg4.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb6fel.ent",
    chain_id = "A",
    residue_number=129,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = glutamate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/6fel.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb6gkf.ent",
    chain_id = "A",
    residue_number=129,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = glutamate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/6gkf.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb6gkg.ent",
    chain_id = "A",
    residue_number=129,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = glutamate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/6gkg.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb6sad.ent",
    chain_id = "A",
    residue_number=129,
    insertion_code = None,
    wildtype_amino_acid = aspartate,
    variant_amino_acid = glutamate,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/6sad.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5bsk.ent",
    chain_id = "A",
    residue_number=48,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5bsk.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb5w8v.ent",
    chain_id = "A",
    residue_number=48,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/5w8v.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb6bnj.ent",
    chain_id = "A",
    residue_number=48,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/6bnj.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1bzy.ent",
    chain_id = "A",
    residue_number=67,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1bzy.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb1d6n.ent",
    chain_id = "A",
    residue_number=67,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/1d6n.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
    radius= 10.0,
    distance_cutoff= 4.5,
))

queries.add(SingleResidueVariantAtomicQuery(
    pdb_path = "/home/gayatrir/DATA/pdb/pdb3ggc.ent",
    chain_id = "A",
    residue_number=67,
    insertion_code = None,
    wildtype_amino_acid = leucine,
    variant_amino_acid = phenylalanine,
    pssm_paths = {"A": "/mnt/csb/DeepRank-Mut-DATA/pssm_update/3ggc.A.pdb.pssm"},
    targets={targets.BINARY: 1.0},
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
