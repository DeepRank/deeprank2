## residue core features
RESTYPE = "res_type" # AminoAcid object; former FEATURENAME_AMINOACID
RESCHARGE = "res_charge" # float(<0); former FEATURENAME_CHARGE (was not assigned) 
RESSIZE = "res_size" # int; former FEATURENAME_SIZE
POLARITY = "polarity" #  Polarity object; former FEATURENAME_POLARITY
HBDONORS = "hb_donors" # int; former FEATURENAME_HYDROGENBONDDONORS
HBACCEPTORS = "hb_acceptors"# int; former FEATURENAME_HYDROGENBONDACCEPTORS

# atomic element core features
ATOMTYPE = "atom_type" # AtomicElement object
ATOMCHARGE = "atom_charge" # float
VANDERWAALSPARAMETERS = "vanderwaals_parameters" # ?
PDBOCCUPANCY = "pdb_occupancy" # float(0 < x < 1)

## conservation features
PSSM = "pssm" # list[20xint]; former FEATURENAME_PSSM
INFOCONTENT = "info_content" # float; former FEATURENAME_INFORMATIONCONTENT
CONSERVATION = "conservation" # int; former FEATURENAME_PSSMWILDTYPE

## variant features
VARIANTRES = "variant_res" # AminoAcid object; former FEATURENAME_VARIANTAMINOACID
DIFFCHARGE = "diff_charge" # float
DIFFSIZE = "diff_size" # int; former FEATURENAME_SIZEDIFFERENCE
DIFFPOLARITY = "diff_polarity" # [type?]; former FEATURENAME_POLARITYDIFFERENCE
DIFFHBDONORS = "diff_hb_donors" # int; former FEATURENAME_HYDROGENBONDDONORSDIFFERENCE
DIFFHBACCEPTORS = "diff_hb_acceptors" # int; former FEATURENAME_HYDROGENBONDACCEPTORSDIFFERENCE
DIFFCONSERVATION = "diff_conservation" # int; former FEATURENAME_PSSMDIFFERENCE & FEATURENAME_CONSERVATIONDIFFERENCE

## protein context features
BSA = "bsa" # float; former FEATURENAME_BURIEDSURFACEAREA
HSE = "hse" # list[3xfloat]; former FEATURENAME_HALFSPHEREEXPOSURE 
SASA = "sasa" # float; former FEATURENAME_SASA
RESDEPTH = "res_depth" # float; former FEATURENAME_RESIDUEDEPTH
