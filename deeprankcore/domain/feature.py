# node features

## generic features
FEATURE_NODE_MAINCHAIN = "main_chain" # bool; previously: FEATURENAME_CHAIN
FEATURE_NODE_POSITION = "coordinates" # list[3xfloat]; previously: FEATURENAME_POSITION

## residue core features
FEATURE_NODE_RESTYPE = "res_type" # AminoAcid object; previously: FEATURENAME_AMINOACID
FEATURE_NODE_RESCHARGE = "res_charge" # float(<0)
FEATURE_NODE_POLARITY = "polarity" # FEATURENAME_POLARITY = "polarity" : Polarity object
FEATURE_NODE_RESIDUESIZE = "residue_size" # FEATURENAME_SIZE = "size" : int --> called it residue size in case we will also have an atom size in the future
FEATURE_NODE_HBONDDONORS = "hbond_donors" # FEATURENAME_HYDROGENBONDDONORS = "hb_donors" : int
FEATURE_NODE_HBONDACCEPTORS = "hbond_acceptors"# FEATURENAME_HYDROGENBONDACCEPTORS = "hb_acceptors" : int

## conservation features
FEATURE_NODE_PSSM = "pssm" # FEATURENAME_PSSM = "pssm": list[20xint]
FEATURE_NODE_INFORMATIONCONTENT = "information_content" # FEATURENAME_INFORMATIONCONTENT = "ic" : float
FEATURE_NODE_CONSERVATION = "conservation" # FEATURENAME_PSSMWILDTYPE = "pssm_wildtype" : int

## protein context features
FEATURE_NODE_BURIEDSURFACEAREA = "buried_surface_area" # FEATURENAME_BURIEDSURFACEAREA = "bsa": float
FEATURE_NODE_HALFSPHEREEXPOSURE = "half_sphere_exposure" # FEATURENAME_HALFSPHEREEXPOSURE = "hse":  list[3xfloat] 
FEATURE_NODE_RESIDUEDEPTH = "residue_depth" # FEATURENAME_RESIDUEDEPTH = "depth" : float
FEATURE_NODE_SASA = "sasa" # FEATURENAME_SASA = "sasa" : float

## variant features
FEATURE_NODE_VARIANTRESIDUE = "variant_residue" # FEATURENAME_VARIANTAMINOACID = "variant": AminoAcid object
FEATURE_NODE_DIFFERENCESIZE = "difference_size" # FEATURENAME_SIZEDIFFERENCE = "size_difference" : int
FEATURE_NODE_DIFFERENCEPOLARITY = "difference_polarity" # FEATURENAME_POLARITYDIFFERENCE = "polarity_difference"
FEATURE_NODE_DIFFERENCEHBONDDONORS = "difference_hbond_donors" # FEATURENAME_HYDROGENBONDDONORSDIFFERENCE = "hb_donors_difference"
FEATURE_NODE_DIFFERENCEHBONDACCEPTORS = "difference_hbond_acceptors" # FEATURENAME_HYDROGENBONDACCEPTORSDIFFERENCE = "hb_acceptors_difference"
FEATURE_NODE_DIFFERENCECHARGE = "difference_charge" # if we include CHARGE, it probably makes sense to include this one too
FEATURE_NODE_VARIANTCONSERVATION = "variant_conservation" # FEATURENAME_PSSMVARIANT = "pssm_variant" : int  --> isn't this information basically covered by FEATURENAME_NODE_CONSERVATIONDIFFERENCE? If not, then we should maybe also include VARIANT versions for: SIZE, POLARITY, HBONDDONORS, HBONDACCEPTORS
FEATURE_NODE_DIFFERENCECONSERVATION = "difference_conservation" # FEATURENAME_PSSMDIFFERENCE = "pssm_difference" : int
# FEATURENAME_CONSERVATION = "conservation" # int; unused; i think superceded by FEATURENAME_PSSMVARIANT
# FEATURENAME_CONSERVATIONDIFFERENCE = "conservation_difference" # unused, i think superceded by: FEATURENAME_PSSMDIFFERENCE



# edge features
FEATURE_EDGE_COVALENT = "covalent" # FEATURENAME_COVALENT = "covalent" : bool
FEATURE_EDGE_ELECTROSTATIC = "electrostatic" # FEATURENAME_EDGECOULOMB = "coulomb"
FEATURE_EDGE_VANDERWAALS = "vanderwaals" # FEATURENAME_EDGEVANDERWAALS = "vanderwaals"
FEATURE_EDGE_DISTANCE = "distance" # FEATURENAME_EDGEDISTANCE = "dist"
# FEATURE_EDGE_INTERACTIONTYPE = "interaction_type" # FEATURENAME_EDGETYPE = "type" --> replace by CIS/SAMECHAIN
FEATURE_EDGE_CIS = "cis" # FEATURENAME_EDGESAMECHAIN = "same_chain" # bool --> unused; I think superceded by INTERACTIONTYPE, but this parameter makes more sense to me

# ## edge types --> use EDGE_CIS instead?
# FEATURE_EDGE_CISINTERACTION = "cis_interaction" # EDGETYPE_INTERNAL = "internal"
# FEATURE_EDGE_TRANSINTERACTION = "trans_interaction"# EDGETYPE_INTERFACE = "interface"
