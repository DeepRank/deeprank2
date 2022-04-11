# %matplotlib inline
from deeprank_gnn.tools.graph import hdf5_to_graph, plotly_3d, plotly_2d


def tsne_graph(grp, method):

    import plotly.offline as py # pylint: disable=import-outside-toplevel

    py.init_notebook_mode(connected=True)

    g = hdf5_to_graph(grp)

    plotly_2d(g, offline=True, iplot=False, method=method)


def graph3d(grp):

    import plotly.offline as py # pylint: disable=import-outside-toplevel

    py.init_notebook_mode(connected=True)

    g = hdf5_to_graph(grp)

    plotly_3d(g, offline=True, iplot=False)
