## generic features
MAINCHAIN = "main_chain" # bool; previously: FEATURENAME_CHAIN
POSITION = "coordinates" # list[3xfloat]; previously: FEATURENAME_POSITION

## residue core features
RESTYPE = "res_type" # AminoAcid object; previously: FEATURENAME_AMINOACID
RESCHARGE = "res_charge" # float(<0)
POLARITY = "polarity" # FEATURENAME_POLARITY = "polarity" : Polarity object
RESSIZE = "res_size" # FEATURENAME_SIZE = "size" : int --> called it residue size in case we will also have an atom size in the future
HBONDDONORS = "hbond_donors" # FEATURENAME_HYDROGENBONDDONORS = "hb_donors" : int
HBONDACCEPTORS = "hbond_acceptors"# FEATURENAME_HYDROGENBONDACCEPTORS = "hb_acceptors" : int

## conservation features
PSSM = "pssm" # FEATURENAME_PSSM = "pssm": list[20xint]
INFOCONTENT = "info_content" # FEATURENAME_INFORMATIONCONTENT = "ic" : float
CONSERVATION = "conservation" # FEATURENAME_PSSMWILDTYPE = "pssm_wildtype" : int

## protein context features
BSA = "bsa" # FEATURENAME_BURIEDSURFACEAREA = "bsa": float
HSE = "hse" # FEATURENAME_HALFSPHEREEXPOSURE = "hse":  list[3xfloat] 
SASA = "sasa" # FEATURENAME_SASA = "sasa" : float
RES_DEPTH = "res_depth" # FEATURENAME_RESIDUEDEPTH = "depth" : float

## variant features
VARIANTRES = "variant_res" # FEATURENAME_VARIANTAMINOACID = "variant": AminoAcid object
VARIANTCONSERVATION = "variant_conservation" # FEATURENAME_PSSMVARIANT = "pssm_variant" : int  --> isn't this information basically covered by FEATURENAME_NODE_CONSERVATIONDIFFERENCE? If not, then we should maybe also include VARIANT versions for: SIZE, POLARITY, HBONDDONORS, HBONDACCEPTORS
DIFFCONSERVATION = "diff_conservation" # FEATURENAME_PSSMDIFFERENCE = "pssm_difference" : int
DIFFSIZE = "diff_size" # FEATURENAME_SIZEDIFFERENCE = "size_difference" : int
DIFFPOLARITY = "diff_polarity" # FEATURENAME_POLARITYDIFFERENCE = "polarity_difference"
DIFFHBONDDONORS = "diff_hbond_donors" # FEATURENAME_HYDROGENBONDDONORSDIFFERENCE = "hb_donors_difference"
DIFFHBONDACCEPTORS = "diff_hbond_acceptors" # FEATURENAME_HYDROGENBONDACCEPTORSDIFFERENCE = "hb_acceptors_difference"
DIFFCHARGE = "diff_charge" # if we include CHARGE, it probably makes sense to include this one too
# FEATURENAME_CONSERVATION = "conservation" # int; unused; i think superceded by FEATURENAME_PSSMVARIANT
# FEATURENAME_CONSERVATIONDIFFERENCE = "conservation_difference" # unused, i think superceded by: FEATURENAME_PSSMDIFFERENCE
