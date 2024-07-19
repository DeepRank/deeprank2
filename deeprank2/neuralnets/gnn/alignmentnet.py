import torch
from torch import nn

# ruff: noqa: ANN001, ANN201

__author__ = "Daniel-Tobias Rademaker"


class GNNLayer(nn.Module):
    """Custom-defined layer of a Graph Neural Network.

    Args:
        nmb_edge_projection: Number of features in the edge projection.
        nmb_hidden_attr: Number of features in the hidden attributes.
        nmb_output_features: Number of output features.
        message_vector_length: Length of the message vector.
        nmb_mlp_neurons: Number of neurons in the MLP.
        act_fn: Activation function. Defaults to nn.SiLU().
        is_last_layer: Whether this is the last layer of the GNN. Defaults to True.
    """

    def __init__(
        self,
        nmb_edge_projection,
        nmb_hidden_attr,
        nmb_output_features,
        message_vector_length,
        nmb_mlp_neurons,
        act_fn=nn.SiLU(),  # noqa: B008
        is_last_layer=True,
    ):
        super().__init__()
        # The MLP that takes in atom-pairs and creates the Mij's
        self.edge_mlp = nn.Sequential(
            nn.Linear(nmb_edge_projection + nmb_hidden_attr * 2, nmb_mlp_neurons),
            act_fn,
            nn.Linear(nmb_mlp_neurons, message_vector_length),
            act_fn,
        )

        # The node-MLP, creates a new node-representation given the Mi's
        self.node_mlp = nn.Sequential(
            nn.BatchNorm1d(message_vector_length + nmb_hidden_attr),
            nn.Linear(message_vector_length + nmb_hidden_attr, nmb_mlp_neurons),
            act_fn,
            nn.Linear(nmb_mlp_neurons, nmb_mlp_neurons),
            act_fn,
            nn.Linear(nmb_mlp_neurons, nmb_hidden_attr),
        )

        # Only last layer have attention and output modules
        if is_last_layer:
            # attention mlp, to weight the ouput significance
            self.attention_mlp = nn.Sequential(
                nn.Linear(nmb_hidden_attr, nmb_mlp_neurons),
                act_fn,
                nn.Linear(nmb_mlp_neurons, 1),
                nn.Sigmoid(),
            )

            # Create the output vector per node we are interested in
            self.output_mlp = nn.Sequential(
                nn.Linear(nmb_hidden_attr, nmb_mlp_neurons),
                act_fn,
                nn.Linear(nmb_mlp_neurons, nmb_output_features),
            )

    # MLP that takes in the node-attributes of nodes (source + target), the edge attributes
    # and node attributes in order to create a 'message vector'between those
    # nodes
    def edge_model(self, edge_attr, hidden_features_source, hidden_features_target):
        cat = torch.cat([edge_attr, hidden_features_source, hidden_features_target], dim=1)
        return self.edge_mlp(cat)

    # A function that updates the node-attributes. Assumed that submessages
    # are already summed
    def node_model(self, summed_edge_message, hidden_features):
        cat = torch.cat([summed_edge_message, hidden_features], dim=1)
        output = self.node_mlp(cat)
        return hidden_features + output

    # Sums the individual sub-messages (multiple per node) into singel message
    # vector per node
    def sum_messages(self, edges, messages, nmb_nodes):
        row, _ = edges
        summed_messages_shape = (nmb_nodes, messages.size(1))
        result = messages.new_full(summed_messages_shape, 0)
        row = row.unsqueeze(-1).expand(-1, messages.size(1))
        result.scatter_add_(0, row, messages)
        return result

    # Runs the GNN
    # steps is number of times it exanges info with neighbors
    def update_nodes(self, edges, edge_attr, hidden_features, steps=1):
        (
            row,
            col,
        ) = edges  # a single edge is defined as the index of atom1 and the index of atom2
        h = hidden_features  # shortening the variable name
        # It is possible to run input through the same same layer multiple
        # times
        for _ in range(steps):
            node_pair_messages = self.edge_model(edge_attr, h[row], h[col])  # get all atom-pair messages
            # sum all messages per node to single message vector
            messages = self.sum_messages(edges, node_pair_messages, len(h))
            # Use the messages to update the node-attributes
            h = self.node_model(messages, h)
        return h

    # output, every node creates a prediction + an estimate how sure it is of
    # its prediction. Only done by last 'GNN layer'
    def output(self, hidden_features, get_attention=True):
        output = self.output_mlp(hidden_features)
        if get_attention:
            return output, self.attention_mlp(hidden_features)
        return output


class SuperGNN(nn.Module):
    """SuperGNN is a class that defines multiple GNN layers.

    In particular, the `preproc_edge_mlp` and `preproc_node_mlp` are meant to
    preprocess the edge and node attributes, respectively.

    The `modlist` is a list of GNNLayer objects.

    Args:
        nm_edge_attr: Number of edge features.
        nmb_node_attr: Number of node features.
        nmb_hidden_attr: Number of hidden features.
        nmb_mlp_neurons: Number of neurons in the MLP.
        nmb_edge_projection: Number of edge projections.
        nmb_gnn_layers: Number of GNN layers.
        nmb_output_features: Number of output features.
        message_vector_length: Length of the message vector.
        act_fn: Activation function. Defaults to nn.SiLU().
    """

    def __init__(
        self,
        nmb_edge_attr,
        nmb_node_attr,
        nmb_hidden_attr,
        nmb_mlp_neurons,
        nmb_edge_projection,
        nmb_gnn_layers,
        nmb_output_features,
        message_vector_length,
        act_fn=nn.SiLU(),  # noqa: B008
    ):
        super().__init__()

        # Since edge_atributes go into every layer, it might be betetr to learn
        # a better/smarter representation of them first
        self.preproc_edge_mlp = nn.Sequential(
            nn.BatchNorm1d(nmb_edge_attr),
            nn.Linear(nmb_edge_attr, nmb_mlp_neurons),
            nn.BatchNorm1d(nmb_mlp_neurons),
            act_fn,
            nn.Linear(nmb_mlp_neurons, nmb_edge_projection),
            act_fn,
        )

        # Project the node_attributes to the same size as the hidden vector
        self.preproc_node_mlp = nn.Sequential(
            nn.BatchNorm1d(nmb_node_attr),
            nn.Linear(nmb_node_attr, nmb_mlp_neurons),
            nn.BatchNorm1d(nmb_mlp_neurons),
            act_fn,
            nn.Linear(nmb_mlp_neurons, nmb_hidden_attr),
            act_fn,
        )

        self.modlist = nn.ModuleList(
            [
                GNNLayer(
                    nmb_edge_projection,
                    nmb_hidden_attr,
                    nmb_output_features,
                    message_vector_length,
                    nmb_mlp_neurons,
                    is_last_layer=(gnn_layer == (nmb_gnn_layers - 1)),
                )
                for gnn_layer in range(nmb_gnn_layers)
            ],
        )

    # always use this function before running the GNN layers
    def preprocess(self, edge_attr, node_attr):
        edge_attr = self.preproc_edge_mlp(edge_attr)
        hidden_features = self.preproc_node_mlp(node_attr)
        return edge_attr, hidden_features

    # Runs data through layers and return output. Potentially, attention can
    # also be returned
    def run_through_network(self, edges, edge_attr, node_attr, with_output_attention=False):
        edge_attr, node_attr = self.preprocess(edge_attr, node_attr)
        for layer in self.modlist:
            node_attr = layer.update_nodes(edges, edge_attr, node_attr)
        if with_output_attention:
            representations, attention = self.modlist[-1].output(node_attr, True)  # (boolean-positional-value-in-call)
            return representations, attention
        return self.modlist[-1].output(node_attr, True)  # (boolean-positional-value-in-call)


class AlignmentGNN(SuperGNN):
    """Architecture based on multiple :class:`GNNLayer` layers, suited for both regression and classification tasks.

    It applies different layers to the nodes and edges of a graph (`preproc_edge_mlp` and `preproc_node_mlp`),
    and then applies multiple GNN layers (`modlist`).

    Args:
        nm_edge_attr: Number of edge features.
        nmb_node_attr: Number of node features.
        nmb_output_features: Number of output features.
        nmb_hidden_attr: Number of hidden features.
        message_vector_length: Length of the message vector.
        nmb_mlp_neurons: Number of neurons in the MLP.
        nmb_gnn_layers: Number of GNN layers.
        nmb_edge_projection: Number of edge projections.
        act_fn: Activation function. Defaults to nn.SiLU().

    """

    def __init__(
        self,
        nmb_edge_attr,
        nmb_node_attr,
        nmb_output_features,
        nmb_hidden_attr,
        message_vector_length,
        nmb_mlp_neurons,
        nmb_gnn_layers,
        nmb_edge_projection,
        act_fn=nn.SiLU(),  # noqa: B008
    ):
        super().__init__(
            nmb_edge_attr,
            nmb_node_attr,
            nmb_hidden_attr,
            nmb_mlp_neurons,
            nmb_edge_projection,
            nmb_gnn_layers,
            nmb_output_features,
            message_vector_length,
            act_fn,
        )

    # Run over all layers, and return the ouput vectors
    def forward(self, edges, edge_attr, node_attr):
        return self.run_through_network(edges, edge_attr, node_attr)
