import plotly.offline as py

from deeprankcore.tools.visualization.plotting import (hdf5_to_networkx,
                                                       plotly_2d, plotly_3d)


def tsne_graph(grp, method):

    py.init_notebook_mode(connected=True)

    g = hdf5_to_networkx(grp)

    plotly_2d(g, offline=True, iplot=False, method=method)


def graph3d(grp):

    py.init_notebook_mode(connected=True)

    g = hdf5_to_networkx(grp)

    plotly_3d(g, offline=True, iplot=False)
