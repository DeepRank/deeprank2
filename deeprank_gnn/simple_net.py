import logging

import torch
from torch.nn import Parameter, Module, Linear
from torch.nn.functional import softmax, leaky_relu, relu
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import max_pool_x
from torch_geometric.data import DataLoader


_log = logging.getLogger(__name__)


class SimpleMessageLayer(Module):

    def __init__(self,
                 count_node_features,
                 count_edge_features):

        super(SimpleMessageLayer, self).__init__()

        self._message_size = 16

        self._fe = Linear(2 * count_node_features + count_edge_features, self._message_size)
        self._fh = Linear(count_node_features + self._message_size, count_node_features)

    def forward(self, node_features, edge_node_indices, edge_features):

        node1_indices, node2_indices = edge_node_indices
        count_nodes = len(node_features)

        if edge_features.dim() == 1:
            edge_features = edge_features.unsqueeze(-1)

        node1_features = node_features[node1_indices]
        node2_features = node_features[node2_indices]

        message_input = torch.cat([node1_features, node2_features, edge_features], dim=1)
        messages_per_edge = self._fe(message_input)

        message_factors_per_edge = softmax(leaky_relu(messages_per_edge), dim=1)

        message_sums_per_node = scatter_sum(message_factors_per_edge, node1_indices, dim=0)

        node_input = torch.cat([node_features, message_sums_per_node], dim=1)
        z = self._fh(node_input)

        return z


class SimpleNetwork(Module):

    number_of_message_layers = 2

    def __init__(self, input_shape, output_shape, input_shape_edge):
        """
            Args:
                input_shape(int): number of node input features
                output_shape(int): number of output value per graph
                input_shape_edge(int): number of edge input features
        """

        super(SimpleNetwork, self).__init__()

        self._count_message_layers = SimpleNetwork.number_of_message_layers

        self._message_layers_internal = []
        for layer_index in range(self._count_message_layers):
            self._message_layers_internal.append(SimpleMessageLayer(input_shape, input_shape_edge))

        self._fc = Linear(input_shape, output_shape)

    def forward(self, data):

        node_features_internal = data.x.clone().detach()
        for layer_index in range(self._count_message_layers):

            node_features_internal = self._message_layers_internal[layer_index](node_features_internal, data.internal_edge_index, data.internal_edge_attr)

        batch = data.batch

        node_features_internal = scatter_mean(node_features_internal, batch, dim=0)

        z = relu(self._fc(node_features_internal))

        return z
