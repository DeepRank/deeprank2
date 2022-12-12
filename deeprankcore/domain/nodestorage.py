## main group
NODE = "node_features"

## metafeatures
NAME = "_name"
CHAINID = "_chain_id" # str; former FEATURENAME_CHAIN (was not assigned, but supposedly numeric, now a str)
POSITION = "_position" # list[3xfloat]; former FEATURENAME_POSITION

## residue core features
RESTYPE = "res_type" # AminoAcid object; former FEATURENAME_AMINOACID
RESCHARGE = "charge" # float(<0); former FEATURENAME_CHARGE (was not assigned) 
POLARITY = "polarity" #  Polarity object; former FEATURENAME_POLARITY
RESSIZE = "res_size" # int; former FEATURENAME_SIZE
RESMASS = "res_mass"
RESPI = "res_pI"
HBDONORS = "hb_donors" # int; former FEATURENAME_HYDROGENBONDDONORS
HBACCEPTORS = "hb_acceptors"# int; former FEATURENAME_HYDROGENBONDACCEPTORS

## variant residue features
VARIANTRES = "variant_res" # AminoAcid object; former FEATURENAME_VARIANTAMINOACID
DIFFCHARGE = "diff_charge" # float
DIFFSIZE = "diff_size" # int; former FEATURENAME_SIZEDIFFERENCE
DIFFMASS = "diff_mass"
DIFFPI = "diff_pI"
DIFFPOLARITY = "diff_polarity" # [type?]; former FEATURENAME_POLARITYDIFFERENCE
DIFFHBDONORS = "diff_hb_donors" # int; former FEATURENAME_HYDROGENBONDDONORSDIFFERENCE
DIFFHBACCEPTORS = "diff_hb_acceptors" # int; former FEATURENAME_HYDROGENBONDACCEPTORSDIFFERENCE
DIFFCONSERVATION = "diff_conservation" # int; former FEATURENAME_PSSMDIFFERENCE & FEATURENAME_CONSERVATIONDIFFERENCE

## protein context features
BSA = "bsa" # float; former FEATURENAME_BURIEDSURFACEAREA
HSE = "hse" # list[3xfloat]; former FEATURENAME_HALFSPHEREEXPOSURE 
SASA = "sasa" # float; former FEATURENAME_SASA
RESDEPTH = "res_depth" # float; former FEATURENAME_RESIDUEDEPTH

## conservation features
PSSM = "pssm" # list[20xint]; former FEATURENAME_PSSM
INFOCONTENT = "info_content" # float; former FEATURENAME_INFORMATIONCONTENT
CONSERVATION = "conservation" # int; former FEATURENAME_PSSMWILDTYPE

## atom core features
ATOMTYPE = "atom_type"
ATOMCHARGE = "atom_charge"
PDBOCCUPANCY = "pdb_occupancy"
VDWPARAMETERS = "vdw_parameters"
