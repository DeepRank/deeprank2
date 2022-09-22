# node features
FEATURENAME_NODE_COVALENT = "covalent" # bool       # FEATURENAME_COVALENT = "covalent"
FEATURENAME_NODE_COORDINATES = "coordinates" # list[3xfloat]      # FEATURENAME_POSITION = "pos"
FEATURENAME_NODE_RESIDUE = "residue" # AminoAcid object     # FEATURENAME_AMINOACID = "type"
FEATURENAME_NODE_VARIANTRESIDUE = "variant_residue" # AminoAcid object      # FEATURENAME_VARIANTAMINOACID = "variant"
# FEATURENAME_CHAIN = "chain" # bool; I think unused. It is checked twice, but never assigned. Not clear what is meant by "chain"
# FEATURENAME_CHARGE = "charge" # float(<0); unused, but the information is present in domain.aminoacid.py; any particular reason why it is not used?
FEATURENAME_NODE_POLARITY = "polarity" # Polarity object        # FEATURENAME_POLARITY = "polarity"
FEATURENAME_NODE_SIZE = "size" # int        # potentially rename if atom features will also include a size
FEATURENAME_NODE_BURIEDSURFACEAREA = "buried_surface_area" # float      # FEATURENAME_BURIEDSURFACEAREA = "bsa"
FEATURENAME_NODE_HALFSPHEREEXPOSURE = "half_sphere_exposire" # list[3xfloat] # can't find this in the code      # FEATURENAME_HALFSPHEREEXPOSURE = "hse"
FEATURENAME_NODE_PSSM = "pssm" # FEATURENAME_PSSM = "pssm"
# FEATURENAME_CONSERVATION = "conservation"
# FEATURENAME_CONSERVATIONDIFFERENCE = "conservation_difference"
# FEATURENAME_INFORMATIONCONTENT = "ic"
# FEATURENAME_RESIDUEDEPTH = "depth"
# FEATURENAME_PSSMDIFFERENCE = "pssm_difference"
# FEATURENAME_PSSMWILDTYPE = "pssm_wildtype"
# FEATURENAME_PSSMVARIANT = "pssm_variant"
# FEATURENAME_SASA = "sasa"
# FEATURENAME_SIZEDIFFERENCE = "size_difference"
# FEATURENAME_POLARITYDIFFERENCE = "polarity_difference"
# FEATURENAME_HYDROGENBONDDONORS = "hb_donors"
# FEATURENAME_HYDROGENBONDDONORSDIFFERENCE = "hb_donors_difference"
# FEATURENAME_HYDROGENBONDACCEPTORS = "hb_acceptors"
# FEATURENAME_HYDROGENBONDACCEPTORSDIFFERENCE = "hb_acceptors_difference"


# edge features
FEATURENAME_EDGE_ELECTROSTATICPOTENTIAL = "electrostatic_potential" # FEATURENAME_EDGECOULOMB = "coulomb"
FEATURENAME_EDGE_LJPOTENTIAL = "lj_potential" # FEATURENAME_EDGEVANDERWAALS = "vanderwaals"
FEATURENAME_EDGE_DISTANCE = "distance" # FEATURENAME_EDGEDISTANCE = "dist"
FEATURENAME_EDGE_RESIDUES = "residues" # FEATURENAME_EDGETYPE = "type"
# FEATURENAME_EDGESAMECHAIN = "same_chain" #unused; is this planned to be used moving forward?

# feature values
EDGETYPE_CIS = "cis_interaction" # EDGETYPE_INTERNAL = "internal"
EDGETYPE_TRANS = "trans_interaction"# EDGETYPE_INTERFACE = "interface"
