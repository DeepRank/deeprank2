# %matplotlib inline

import community
import networkx as nx
import torch

from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

from torch_geometric.data import Data
from deeprank_gnn.community_pooling import *
from deeprank_gnn.tools.visualization import hdf5_to_networkx, plotly_3d, plotly_2d


def tsne_graph(grp, method):

    import plotly.offline as py # pylint: disable=import-outside-toplevel

    py.init_notebook_mode(connected=True)

    g = hdf5_to_networkx(grp)

    plotly_2d(g, offline=True, iplot=False, method=method)


def graph3d(grp):

    import plotly.offline as py # pylint: disable=import-outside-toplevel

    py.init_notebook_mode(connected=True)

    g = hdf5_to_networkx(grp)

    plotly_3d(g, offline=True, iplot=False)
