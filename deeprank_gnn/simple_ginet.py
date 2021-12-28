import logging

import torch
from torch.nn import Parameter, Module, Linear
from torch.nn.functional import softmax, leaky_relu, relu, dropout
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn.inits import uniform

_log = logging.getLogger(__name__)


class SimpleGiMessageLayer(Module):

    def __init__(self,
                 count_node_features,
                 count_edge_features, node_output_size):

        super(SimpleGiMessageLayer, self).__init__()

        edge_output_size = count_edge_features

        self._fn = Linear(count_node_features, node_output_size)
        uniform(count_node_features, self._fn.weight)

        self._fe = Linear(count_edge_features, edge_output_size)
        uniform(count_edge_features, self._fe.weight)

        attention_input_size = 2 * node_output_size + edge_output_size
        self._fa = Linear(attention_input_size, 1)
        uniform(attention_input_size, self._fa.weight)

    def forward(self, node_features, edge_node_indices, edge_features):

        node0_indices, node1_indices = edge_node_indices

        count_nodes = len(node_features)

        if edge_features.dim() == 1:
            edge_features = edge_features.unsqueeze(-1)

        node0_output = self._fn(node_features[node0_indices])
        node1_output = self._fn(node_features[node1_indices])

        edge_output = self._fe(edge_features)

        attention_input = torch.cat([node0_output, node1_output, edge_output], dim=1)
        attention_output = self._fa(attention_input)
        attention = softmax(leaky_relu(attention_output), dim=1)

        attenuated_node_output = attention * node0_output

        z = scatter_sum(attenuated_node_output, node0_indices, dim=0)
        return z


class SimpleGiNetwork(Module):

    number_of_message_layers = 2

    def __init__(self, input_shape, output_shape, input_shape_edge):
        """
            Args:
                input_shape(int): number of node input features
                output_shape(int): number of output value per graph
                input_shape_edge(int): number of edge input features
        """

        super(SimpleGiNetwork, self).__init__()

        self._message_layer1 = SimpleGiMessageLayer(input_shape, input_shape_edge, 16)
        self._message_layer2 = SimpleGiMessageLayer(16, input_shape_edge, 32)

        self._fc1 = Linear(32, 128)
        uniform(32, self._fc1.weight)

        self._fc2 = Linear(128, output_shape)
        uniform(128, self._fc2.weight)

    def forward(self, data):

        updated1 = relu(self._message_layer1(data.x, data.internal_edge_index, data.internal_edge_attr))
        updated2 = relu(self._message_layer2(updated1, data.internal_edge_index, data.internal_edge_attr))

        updated_per_graph = scatter_mean(updated2, data.batch, dim=0)

        output1 = dropout(relu(self._fc1(updated_per_graph)), 0.4, training=self.training)

        output2 = self._fc2(output1)

        return output2
