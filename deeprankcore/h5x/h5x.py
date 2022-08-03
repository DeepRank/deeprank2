#!/usr/bin/env python

import os
from h5xplorer.h5xplorer import h5xplorer
from deeprankcore.h5x.h5x_menu import context_menu

baseimport = os.path.dirname(
    os.path.abspath(__file__)) + "/baseimport.py"
app = h5xplorer(context_menu,
                baseimport=baseimport, extended_selection=False)
