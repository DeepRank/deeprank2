from abc import ABC, abstractmethod
from typing import Dict

import numpy

from deeprank_gnn.models.graph import Node


class NodeFeatureProvider(ABC):
    "implementations of this class should provide node features"

    @abstractmethod
    def get_features(self, edge: Node) -> Dict[str, numpy.ndarray]:
        """ Each implementation of this method should provide a dictionary,
            containing the feature value per feature name for this node.
        """
        pass
