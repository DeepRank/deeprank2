## generic features
MAINCHAIN = "main_chain" # bool; former FEATURENAME_CHAIN
POSITION = "position" # list[3xfloat]; former FEATURENAME_POSITION

## residue core features
RESTYPE = "res_type" # AminoAcid object; former FEATURENAME_AMINOACID
RESCHARGE = "res_charge" # float(<0); former FEATURE_NODE_RESCHARGE
POLARITY = "polarity" #  Polarity object; former FEATURE_NODE_POLARITY
RESSIZE = "res_size" # int; former FEATURENAME_SIZE
HBDONORS = "hb_donors" # int; former FEATURENAME_HYDROGENBONDDONORS
HBACCEPTORS = "hb_acceptors"# int; former FEATURENAME_HYDROGENBONDACCEPTORS

## variant features
VARIANTRES = "variant_res" # AminoAcid object; former FEATURENAME_VARIANTAMINOACID
VARIANTCONSERVATION = "variant_conservation" # int; former FEATURENAME_PSSMVARIANT
DIFFCONSERVATION = "diff_conservation" # int; former FEATURENAME_PSSMDIFFERENCE
DIFFSIZE = "diff_size" # int; former FEATURENAME_SIZEDIFFERENCE
DIFFPOLARITY = "diff_polarity" # [type?]; former FEATURENAME_POLARITYDIFFERENCE
DIFFHBDONORS = "diff_hb_donors" # int; former FEATURENAME_HYDROGENBONDDONORSDIFFERENCE
DIFFHBACCEPTORS = "diff_hb_acceptors" # int; former FEATURENAME_HYDROGENBONDACCEPTORSDIFFERENCE
DIFFCHARGE = "diff_charge" # if we include CHARGE, it probably makes sense to include this one too

## conservation features
PSSM = "pssm" # list[20xint]; former FEATURENAME_PSSM
INFOCONTENT = "info_content" # float; former FEATURENAME_INFORMATIONCONTENT
CONSERVATION = "conservation" # int; former FEATURENAME_PSSMWILDTYPE

## protein context features
BSA = "bsa" # float; former FEATURENAME_BURIEDSURFACEAREA
HSE = "hse" # list[3xfloat]; former FEATURENAME_HALFSPHEREEXPOSURE 
SASA = "sasa" # float; former FEATURENAME_SASA
RESDEPTH = "res_depth" # float; former FEATURENAME_RESIDUEDEPTH
