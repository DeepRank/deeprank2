## main group
NODE = "node_features"

## metafeatures
NAME = "_name"
CHAINID = "_chain_id"  # str; former FEATURENAME_CHAIN (was not assigned, but supposedly numeric, now a str)
POSITION = "_position"  # list[3xfloat]; former FEATURENAME_POSITION

## atom core features
ATOMTYPE = "atom_type"
ATOMCHARGE = "atom_charge"
PDBOCCUPANCY = "pdb_occupancy"

## residue core features
RESTYPE = "res_type"  # AminoAcid object; former FEATURENAME_AMINOACID
RESCHARGE = "res_charge"  # float(<0); former FEATURENAME_CHARGE (was not assigned)
POLARITY = "polarity"  #  Polarity object; former FEATURENAME_POLARITY
RESSIZE = "res_size"  # int; former FEATURENAME_SIZE
RESMASS = "res_mass"
RESPI = "res_pI"
HBDONORS = "hb_donors"  # int; former FEATURENAME_HYDROGENBONDDONORS
HBACCEPTORS = "hb_acceptors"  # int; former FEATURENAME_HYDROGENBONDACCEPTORS

## variant residue features
VARIANTRES = "variant_res"  # AminoAcid object; former FEATURENAME_VARIANTAMINOACID
DIFFCHARGE = "diff_charge"  # float
DIFFSIZE = "diff_size"  # int; former FEATURENAME_SIZEDIFFERENCE
DIFFMASS = "diff_mass"
DIFFPI = "diff_pI"
DIFFPOLARITY = "diff_polarity"  # [type?]; former FEATURENAME_POLARITYDIFFERENCE
DIFFHBDONORS = "diff_hb_donors"  # int; former FEATURENAME_HYDROGENBONDDONORSDIFFERENCE
DIFFHBACCEPTORS = "diff_hb_acceptors"  # int; former FEATURENAME_HYDROGENBONDACCEPTORSDIFFERENCE

## conservation features
PSSM = "pssm"  # list[20xint]; former FEATURENAME_PSSM
INFOCONTENT = "info_content"  # float; former FEATURENAME_INFORMATIONCONTENT
CONSERVATION = "conservation"  # int; former FEATURENAME_PSSMWILDTYPE
DIFFCONSERVATION = "diff_conservation"  # int; former FEATURENAME_PSSMDIFFERENCE & FEATURENAME_CONSERVATIONDIFFERENCE

## protein context features
RESDEPTH = "res_depth"  # float; former FEATURENAME_RESIDUEDEPTH
HSE = "hse"  # list[3xfloat]; former FEATURENAME_HALFSPHEREEXPOSURE
SASA = "sasa"  # float; former FEATURENAME_SASA
BSA = "bsa"  # float; former FEATURENAME_BURIEDSURFACEAREA
SECSTRUCT = "sec_struct"  # secondary structure

## inter-residue contacts (IRCs)
IRC_NONNON = "irc_nonpolar_nonpolar"  # int
IRC_NONPOL = "irc_nonpolar_polar"  # int
IRC_NONNEG = "irc_nonpolar_negative"  # int
IRC_NONPOS = "irc_nonpolar_positive"  # int
IRC_POLPOL = "irc_polar_polar"  # int
IRC_POLNEG = "irc_polar_negative"  # int
IRC_POLPOS = "irc_polar_positive"  # int
IRC_NEGNEG = "irc_negative_negative"  # int
IRC_NEGPOS = "irc_negative_positive"  # int
IRC_POSPOS = "irc_positive_positive"  # int
IRCTOTAL = "irc_total"  # int

IRC_FEATURES = [
    IRC_NONNON,
    IRC_NONPOL,
    IRC_NONNEG,
    IRC_NONPOS,
    IRC_POLPOL,
    IRC_POLNEG,
    IRC_POLPOS,
    IRC_NEGNEG,
    IRC_POSPOS,
    IRC_NEGPOS,
    IRCTOTAL,
]
