## Feature groups
NODE = "node_features"
EDGE = "edge_features"

## Metafeatures: non-trainable features starting with '_'
NAMES = "_names"
CHAINID = "_chain_id" # str; former FEATURENAME_CHAIN (was not assigned, but supposedly numeric, now a str)
POSITION = "_position" # list[3xfloat]; former FEATURENAME_POSITION
INDICES = "_indices"
