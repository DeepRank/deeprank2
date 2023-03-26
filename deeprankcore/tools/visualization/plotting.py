import logging
from copy import deepcopy
from typing import Optional

import community
import h5py
import markov_clustering
import matplotlib.pyplot as plt
import networkx
import numpy as np
import plotly.graph_objs as go

from deeprankcore.domain import edgestorage as Efeat
from deeprankcore.domain import nodestorage as Nfeat
from deeprankcore.tools.visualization.embedding import manifold_embedding

_log = logging.getLogger(__name__)


def _get_node_key(value):
    if isinstance(value, str):
        return value

    key = ""
    for item in value:
        if isinstance(item, (bytes, np.bytes_)):
            key = item.decode()

        elif isinstance(item, str):
            key += item

        else:
            key += str(item)

    return key


def hdf5_to_networkx(graph_group: h5py.Group) -> networkx.Graph: # pylint: disable=too-many-locals
    """Read a hdf5 group into a networkx graph."""

    graph = networkx.Graph()

    # read nodes
    node_features_group = graph_group[Nfeat.NODE]
    node_names = [_get_node_key(key) for key in node_features_group[Nfeat.NAME][()]]
    node_features = {}
    node_feature_names = list(node_features_group.keys())
    for node_feature_name in node_feature_names:
        node_features[node_feature_name] = node_features_group[node_feature_name][()]

    for node_index, node_name in enumerate(node_names):
        graph.add_node(node_name)
        for node_feature_name in node_feature_names:
            graph.nodes[node_name][node_feature_name] = node_features[
                node_feature_name
            ][node_index]

    # read edges
    edge_features_group = graph_group[Efeat.EDGE]
    edge_names = edge_features_group[Efeat.NAME][()]
    edge_node_indices = edge_features_group[Efeat.INDEX][()]
    edge_features = {}
    edge_feature_names = list(edge_features_group.keys())
    for edge_feature_name in edge_feature_names:
        edge_features[edge_feature_name] = edge_features_group[edge_feature_name][()]

    for edge_index, _ in enumerate(edge_names):
        node1_index, node2_index = edge_node_indices[edge_index]
        node1_name = node_names[node1_index]
        node2_name = node_names[node2_index]
        edge_key = (node1_name, node2_name)

        graph.add_edge(node1_name, node2_name)
        for edge_feature_name in edge_feature_names:
            graph.edges[edge_key][edge_feature_name] = edge_features[edge_feature_name][
                edge_index
            ]

    return graph


def _get_edge_type_name(value):
    if isinstance(value, (bytes, np.bytes_)):

        return value.decode()

    return value


def plotly_2d( # noqa
    graph: networkx.Graph,
    out: Optional[str] = None,
    offline: bool = False,
    iplot: bool = True,
    disable_plot: bool = False,
    method: str = "louvain",
):
    """Plots the interface graph in 2D."""

    if offline:
        import plotly.offline as py  # pylint: disable=import-outside-toplevel
    else:
        import chart_studio.plotly as py  # pylint: disable=import-outside-toplevel

    pos = np.array(
        [v.tolist() for _, v in networkx.get_node_attributes(graph, Nfeat.POSITION).items()]
    )
    pos2d = manifold_embedding(pos)
    dict_pos = dict(zip(graph.nodes, pos2d))
    networkx.set_node_attributes(graph, dict_pos, "pos2d")

    # remove interface edges for clustering
    gtmp = deepcopy(graph)
    ebunch = []
    for e in graph.edges:
        if graph.edges[e][Efeat.SAMECHAIN] == 0.0:
            ebunch.append(e)
    gtmp.remove_edges_from(ebunch)

    if method == "louvain":
        cluster = community.best_partition(gtmp)

    elif method == "mcl":
        matrix = networkx.to_scipy_sparse_matrix(gtmp)
        # run MCL with default parameters
        result = markov_clustering.run_mcl(matrix)
        mcl_clust = markov_clustering.get_clusters(result)  # get clusters
        cluster = {}
        node_key = list(graph.nodes.keys())
        for ic, c in enumerate(mcl_clust):
            for node in c:
                cluster[node_key[node]] = ic

    # get the colormap for the clsuter line
    ncluster = np.max([v for _, v in cluster.items()]) + 1
    cmap = plt.cm.nipy_spectral
    N = cmap.N
    cmap = [cmap(i) for i in range(N)]
    cmap = cmap[:: int(N / ncluster)]
    cmap = "plasma"

    edge_trace_list, internal_edge_trace_list = [], []

    node_connect = {}
    for edge in graph.edges:

        same_chain = graph.edges[edge[0], edge[1]][Efeat.SAMECHAIN]
        if same_chain == 1.0: # internal
            trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode="lines",
                hoverinfo=None,
                showlegend=False,
                line=go.scatter.Line(color="rgb(110,110,110)", width=3),
            )

        if same_chain == 0.0:  # interface
            trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode="lines",
                hoverinfo=None,
                showlegend=False,
                line=go.scatter.Line(color="rgb(210,210,210)", width=1),
            )
        else:
            continue

        x0, y0 = graph.nodes[edge[0]]["pos2d"]
        x1, y1 = graph.nodes[edge[1]]["pos2d"]

        trace["x"] += (x0, x1, None)
        trace["y"] += (y0, y1, None)

        if same_chain == 1.0: # internal
            internal_edge_trace_list.append(trace)

        if same_chain == 0.0: # interface
            edge_trace_list.append(trace)

        for i in [0, 1]:
            if edge[i] not in node_connect:
                node_connect[edge[i]] = 1
            else:
                node_connect[edge[i]] += 1
    node_trace_a = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="rgb(227,28,28)",
            size=[],
            line=dict(color=[], width=4, colorscale=cmap),
        ),
    )
    # 'rgb(227,28,28)'
    node_trace_b = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="rgb(0,102,255)",
            size=[],
            line=dict(color=[], width=4, colorscale=cmap),
        ),
    )
    # 'rgb(0,102,255)'
    node_trace = [node_trace_a, node_trace_b]

    for x, node in enumerate(graph.nodes):

        index = 0
        if Nfeat.CHAINID in graph.nodes[node]:
            if x == 0:
                first_chain = graph.nodes[node][Nfeat.CHAINID]
            if graph.nodes[node][Nfeat.CHAINID] != first_chain: # This is not very pythonic, but somehow I'm stuck on how to do this without enumerating
                index = 1
        
        pos = graph.nodes[node]["pos2d"]

        node_trace[index]["x"] += (pos[0],)
        node_trace[index]["y"] += (pos[1],)
        node_trace[index]["text"] += (
            "[Clst:" + str(cluster[node]) + "] " + " ".join(node),
        )

        nc = node_connect[node]
        node_trace[index]["marker"]["size"] += (5 + 15 * np.tanh(nc / 5),)
        node_trace[index]["marker"]["line"]["color"] += (cluster[node],)

    fig = go.Figure(
        data=[*internal_edge_trace_list, *edge_trace_list, *node_trace],
        layout=go.Layout(
            title="<br>tSNE connection graph",
            titlefont=dict(size=16),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    if not disable_plot:
        if iplot:
            py.iplot(fig, filename=out)
        else:
            py.plot(fig)


def plotly_3d( # pylint: disable=too-many-locals, too-many-branches # noqa: MC0001
    graph: networkx.Graph,
    out: Optional[str] = None,
    offline: bool = False,
    iplot: bool = True,
    disable_plot: bool = False,
):
    """Plots interface graph in 3D.

    Args:
        graph (:class:`networkx.Graph`): The graph to be plotted.
        out (str, optional): Defaults to None.
        offline (bool, optional): Defaults to False.
        iplot (bool, optional): Defaults to True.
        disable_plot (bool, optional): Defaults to False.
    """

    if offline:
        import plotly.offline as py  # pylint: disable=import-outside-toplevel
    else:
        import chart_studio.plotly as py  # pylint: disable=import-outside-toplevel

    edge_trace_list, internal_edge_trace_list = [], []
    node_connect = {}

    for edge in graph.edges:

        same_chain = graph.edges[edge[0], edge[1]][Efeat.SAMECHAIN]
        if same_chain == 1.0: # internal
            trace = go.Scatter3d(
                x=[],
                y=[],
                z=[],
                text=[],
                mode="lines",
                hoverinfo=None,
                showlegend=False,
                line=go.scatter3d.Line(color="rgb(110,110,110)", width=5),
            )

        elif same_chain == 0.0: # interface
            trace = go.Scatter3d(
                x=[],
                y=[],
                z=[],
                text=[],
                mode="lines",
                hoverinfo=None,
                showlegend=False,
                line=go.scatter3d.Line(color="rgb(210,210,210)", width=2),
            )
        else:
            continue

        x0, y0, z0 = graph.nodes[edge[0]][Nfeat.POSITION]
        x1, y1, z1 = graph.nodes[edge[1]][Nfeat.POSITION]

        trace["x"] += (x0, x1, None)
        trace["y"] += (y0, y1, None)
        trace["z"] += (z0, z1, None)

        if same_chain == 1.0: # internal
            internal_edge_trace_list.append(trace)

        elif same_chain == 0.0: # interface
            edge_trace_list.append(trace)

        for i in [0, 1]:
            if edge[i] not in node_connect:
                node_connect[edge[i]] = 1
            else:
                node_connect[edge[i]] += 1

    node_trace_a = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="rgb(227,28,28)",
            size=[],
            symbol="circle",
            line=dict(color="rgb(50,50,50)", width=2),
        ),
    )

    node_trace_b = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="rgb(0,102,255)",
            size=[],
            symbol="circle",
            line=dict(color="rgb(50,50,50)", width=2),
        ),
    )

    node_trace = [node_trace_a, node_trace_b]

    for x, node in enumerate(graph.nodes):

        index = 0
        if Nfeat.CHAINID in graph.nodes[node]:
            if x == 0:
                first_chain = graph.nodes[node][Nfeat.CHAINID]
            if graph.nodes[node][Nfeat.CHAINID] != first_chain: # This is not very puythonic, but somehow I'm stuck on how to do this without enumerating
                index = 1

        pos = graph.nodes[node][Nfeat.POSITION]

        node_trace[index]["x"] += (pos[0],)
        node_trace[index]["y"] += (pos[1],)
        node_trace[index]["z"] += (pos[2],)
        node_trace[index]["text"] += (" ".join(node),)

        nc = node_connect[node]
        node_trace[index]["marker"]["size"] += (5 + 15 * np.tanh(nc / 5),)

    fig = go.Figure(
        data=[*node_trace, *internal_edge_trace_list, *edge_trace_list],
        layout=go.Layout(
            title="<br>Connection graph",
            titlefont=dict(size=16),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    if not disable_plot:
        if iplot:
            py.iplot(fig, filename=out)
        else:
            py.plot(fig)
