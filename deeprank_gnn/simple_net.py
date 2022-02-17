import logging

import torch
from torch.nn import Parameter, Module, Linear
from torch.nn.functional import softmax, leaky_relu, relu
from torch_scatter import scatter_mean, scatter_sum

_log = logging.getLogger(__name__)


class SimpleMessageLayer(Module):

    def __init__(self,
                 count_node_features,
                 count_edge_features):

        super(SimpleMessageLayer, self).__init__()

        message_size = 4

        # layer for inputting edge, node 0 and node 1
        edge_input_size = 2 * count_node_features + count_edge_features
        self._fe = Linear(edge_input_size, message_size)

        # layer that makes the final output
        node_input_size = count_node_features + message_size
        self._fh = Linear(node_input_size, count_node_features)

    def forward(self, node_features, edge_node_indices, edge_features):

        node0_indices, node1_indices = edge_node_indices

        if edge_features.dim() == 1:
            edge_features = edge_features.unsqueeze(-1)

        node0_features = node_features[node0_indices]
        node1_features = node_features[node1_indices]

        message_input = torch.cat([node0_features, node1_features, edge_features], dim=1)
        messages_per_neighbour = softmax(leaky_relu(self._fe(message_input)), dim=1)

        message_sums_per_node = scatter_sum(messages_per_neighbour, node0_indices, dim=0)

        node_input = torch.cat([node_features, message_sums_per_node], dim=1)
        z = self._fh(node_input)

        return z


class SimpleNetwork(Module):

    def __init__(self, input_shape, output_shape, input_shape_edge):
        """
            Args:
                input_shape(int): number of node input features
                output_shape(int): number of output value per graph
                input_shape_edge(int): number of edge input features
        """

        super(SimpleNetwork, self).__init__()

        self._count_message_layers = 2
        self._count_intermediary_layers = 0

        # The layers that combine edge information per node
        self._message_layers_internal = []
        for layer_index in range(self._count_message_layers):
            self._message_layers_internal.append(SimpleMessageLayer(input_shape, input_shape_edge))

        intermediary_size = 16

        # The layers that convert graph information
        self._fc = Linear(input_shape, intermediary_size)

        self._intermediary_layers = []
        for layer_index in range(self._count_intermediary_layers):
            self._intermediary_layers.append(Linear(intermediary_size, intermediary_size))

        self._fz = Linear(intermediary_size, output_shape)

    def forward(self, data):

        node_features_updated = data.x.clone().detach()
        for layer_index in range(self._count_message_layers):
            node_features_updated = relu(self._message_layers_internal[layer_index](node_features_updated, data.internal_edge_index, data.internal_edge_attr))

        graph_indices = data.batch

        means_per_graph = scatter_mean(node_features_updated, graph_indices, dim=0)

        intermediary = relu(self._fc(means_per_graph))

        for layer_index in range(self._count_intermediary_layers):
            intermediary = relu(self._intermediary_layers[layer_index](intermediary))

        return softmax(relu(self._fz(intermediary)), dim=1)
