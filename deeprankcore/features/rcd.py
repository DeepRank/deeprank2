import logging
import numpy as np
from typing import List
import pdb2sql
from deeprankcore.utils.graph import Node, Graph
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.atom import Atom
from deeprankcore.domain import nodestorage as Nfeat


def count_neighbours(residue):
    pass


def add_features(
    pdb_path: str, 
    graph: Graph,
    distance: float = 5.5,
    *args, **kwargs): # pylint: disable=unused-argument
    
    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
        else:
            raise TypeError(f"Unexpected node type: {type(node.id)}")
    
    

