import logging

import torch
from torch.nn import Parameter, Module, Linear, Sequential, ReLU, Softmax, SiLU
from torch.nn.functional import relu, silu
from torch_scatter import scatter_mean, scatter_sum

_log = logging.getLogger(__name__)


class SimpleConvolutionalLayer(Module):
    def __init__(self, count_node_features, count_edge_features):

        super(SimpleConvolutionalLayer, self).__init__()

        message_size = 4
        hidden_size = 32

        edge_input_size = 2 * count_node_features + count_edge_features
        self._edge_mlp = Sequential(
            Linear(edge_input_size, hidden_size),
            SiLU(),
            Linear(hidden_size, message_size),
            SiLU(),
        )

        node_input_size = count_node_features + message_size
        self._node_mlp = Sequential(
            Linear(node_input_size, hidden_size),
            SiLU(),
            Linear(hidden_size, count_node_features),
            SiLU(),
        )

    def _edge_forward(self, node_features, edge_node_indices, edge_features):

        node0_indices, node1_indices = edge_node_indices

        node0_features = node_features[node0_indices]
        node1_features = node_features[node1_indices]

        message_input = torch.cat(
            [node0_features, node1_features, edge_features], dim=1
        )

        messages_per_neighbour = self._edge_mlp(message_input)

        return messages_per_neighbour

    def _node_forward(self, node_features, message_sums_per_node):

        node_input = torch.cat([node_features, message_sums_per_node], dim=1)

        node_output = self._node_mlp(node_input)

        return node_output

    def forward(self, node_features, edge_node_indices, edge_features):

        node0_indices, node1_indices = edge_node_indices

        messages_per_neighbour = self._edge_forward(
            node_features, edge_node_indices, edge_features
        )

        message_sums_per_node = torch.zeros(
            node_features.shape[0], messages_per_neighbour.shape[1]
        )
        scatter_sum(
            messages_per_neighbour, node0_indices, dim=0, out=message_sums_per_node
        )

        output = self._node_forward(node_features, message_sums_per_node)

        return output


class SimpleNetwork(Module):
    def __init__(self, input_shape, output_shape, input_shape_edge):
        """
        Args:
            input_shape(int): number of node input features
            output_shape(int): number of output value per graph
            input_shape_edge(int): number of edge input features
        """

        super(SimpleNetwork, self).__init__()

        layer_count = 2

        self._internal_convolutional_layers = []
        self._external_convolutional_layers = []
        for layer_index in range(layer_count):
            self._internal_convolutional_layers.append(
                SimpleConvolutionalLayer(input_shape, input_shape_edge)
            )
            self._external_convolutional_layers.append(
                SimpleConvolutionalLayer(input_shape, input_shape_edge)
            )

        hidden_size = 32

        self._graph_mlp = Sequential(
            Linear(2 * input_shape, hidden_size),
            SiLU(),
            Linear(hidden_size, output_shape),
            SiLU(),
        )

    @staticmethod
    def _update(node_features, edge_indices, edge_features, convolutional_layers):

        updated_node_features = node_features.clone().detach()
        for layer_index in range(len(convolutional_layers)):
            updated_node_features = silu(
                convolutional_layers[layer_index](
                    updated_node_features, edge_indices, edge_features
                )
            )

        return updated_node_features

    def _graph_forward(self, features_internal, features_external):

        input_ = torch.cat([features_internal, features_external], dim=1)

        output = self._graph_mlp(input_)
        return output

    def forward(self, data):

        updated_internal = self._update(
            data.x,
            data.internal_edge_index,
            data.internal_edge_attr,
            self._internal_convolutional_layers,
        )
        updated_external = self._update(
            data.x, data.edge_index, data.edge_attr, self._external_convolutional_layers
        )

        graph_indices = data.batch
        means_per_graph_internal = scatter_mean(updated_internal, graph_indices, dim=0)
        means_per_graph_external = scatter_mean(updated_external, graph_indices, dim=0)

        output = self._graph_forward(means_per_graph_internal, means_per_graph_external)
        return output
