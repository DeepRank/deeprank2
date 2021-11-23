import logging

import numpy

from deeprank_gnn.domain.feature import FEATURENAME_EDGETYPE
from deeprank_gnn.domain.graph import EDGETYPE_INTERFACE, EDGETYPE_INTERNAL


_log = logging.getLogger(__name__)


def graph_to_hdf5(graph, hdf5_file):

    graph_group = hdf5_file.create_group(graph.id)

    # store node names
    node_names = numpy.array([str(node) for node in graph.nodes]).astype('S')
    graph_group.create_dataset("nodes", data=node_names)

    # store node features
    node_features_group = graph_group.create_group("node_data")
    node_key_list = list(graph.nodes.keys())
    first_node_data = graph.nodes[node_key_list[0]]
    node_feature_names = list(first_node_data.keys())
    for node_feature_name in node_feature_names:

        node_feature_data = [node_data[node_feature_name] for node_key, node_data in graph.nodes.items()]

        node_features_group.create_dataset(node_feature_name, data=node_feature_data)

    # store edges
    edge_indices = []
    internal_edge_indices = []

    edge_names = []
    internal_edge_names = []

    first_edge_data = list(graph.edges.values())[0]
    edge_feature_names = list([name for name in first_edge_data.keys() if name != FEATURENAME_EDGETYPE])

    edge_feature_data = {name: [] for name in edge_feature_names}
    internal_edge_feature_data = {name: [] for name in edge_feature_names}

    for edge_key, edge_data in graph.edges.items():

        node1, node2 = edge_key
        edge_type = edge_data[FEATURENAME_EDGETYPE]

        node_index1 = node_key_list.index(node1)
        node_index2 = node_key_list.index(node2)

        if edge_type == EDGETYPE_INTERFACE:
            edge_indices.append((node_index1, node_index2))
            edge_names.append(str(edge_key))

        elif edge_type == EDGETYPE_INTERNAL:
            internal_edge_indices.append((node_index1, node_index2))
            internal_edge_names.append(str(edge_key))

        for edge_feature_name in edge_feature_names:
            if edge_type == EDGETYPE_INTERFACE:
                edge_feature_data[edge_feature_name].append(edge_data[edge_feature_name])

            elif edge_type == EDGETYPE_INTERNAL:
                internal_edge_feature_data[edge_feature_name].append(edge_data[edge_feature_name])

    graph_group.create_dataset("edges", data=numpy.array(edge_names).astype('S'))
    graph_group.create_dataset("internal_edges", data=numpy.array(internal_edge_names).astype('S'))

    graph_group.create_dataset("edge_index", data=edge_indices)
    graph_group.create_dataset("internal_edge_index", data=internal_edge_indices)

    edge_feature_group = graph_group.create_group("edge_data")
    internal_edge_feature_group = graph_group.create_group("internal_edge_data")
    for edge_feature_name in edge_feature_names:
        edge_feature_group.create_dataset(edge_feature_name, data=edge_feature_data[edge_feature_name])
        internal_edge_feature_group.create_dataset(edge_feature_name, data=internal_edge_feature_data[edge_feature_name])

    # store targets
    score_group = graph_group.create_group("score")
    for target_name, target_value in graph.targets.items():
        score_group.create_dataset(target_name, data=target_value)
