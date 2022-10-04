NAMES = "_names" # HDF5KEY_GRAPH_NAMES (edge or node??)
COVALENT = "covalent" # FEATURENAME_COVALENT = "covalent" : bool
ELECTROSTATIC = "electrostatic" # FEATURENAME_EDGECOULOMB = "coulomb"
VANDERWAALS = "vanderwaals" # FEATURENAME_EDGEVANDERWAALS = "vanderwaals"
DISTANCE = "distance" # FEATURENAME_EDGEDISTANCE = "dist"
CISTYPE = "cis_type" # FEATURENAME_EDGESAMECHAIN = "same_chain" # bool --> unused; I think superceded by INTERACTIONTYPE, but this parameter makes more sense to me
# FEATURE_EDGE_INTERACTIONTYPE = "interaction_type" # FEATURENAME_EDGETYPE = "type" --> replace by CIS/SAMECHAIN

# ## edge types --> use EDGE_CIS instead?
# FEATURE_EDGE_CISINTERACTION = "cis_interaction" # EDGETYPE_INTERNAL = "internal"
# FEATURE_EDGE_TRANSINTERACTION = "trans_interaction"# EDGETYPE_INTERFACE = "interface"
