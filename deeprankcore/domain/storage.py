# GRAPH GROUPS
HDF5KEY_GRAPH_NODEFEATURES = "node_features"
HDF5KEY_GRAPH_EDGEFEATURES = "edge_features"
HDF5KEY_GRAPH_TARGETVALUES = "target_values"

# GRAPH IDENTIFYERS
HDF5KEY_GRAPH_NAMES = "_names"
HDF5KEY_GRAPH_INDICES = "_indices"

# GRIDS
HDF5KEY_GRID_GRIDPOINTS = "grid_points"
HDF5KEY_GRID_X = "x"
HDF5KEY_GRID_Y = "y"
HDF5KEY_GRID_Z = "z"
HDF5KEY_GRID_CENTER = "center"
HDF5KEY_GRID_MAPPEDFEATURES = "mapped_features"
HDF5KEY_GRID_MAPPEDFEATURESVALUE = "mapped_features_value"



# TO DO LIST:
# - Search/replace outdated nomenclature inside strings: node_data, edge_data, score, edges, nodes, edge_index
#   - is there a way to fuse strings with a given character in between????
# - Update FEATURENAMEs
#   - make sure to search for strings as well
# - Fix clustering groups/names
# - Move above stuff to deeprankcore.domain.feature instead?
# - Is grid stuff relevant here (or is it a CNN thing)?
# - Check featurename changes suggested by Giulia
#   - Features that Giulia suggested to exclude; how about assigning them instead?
