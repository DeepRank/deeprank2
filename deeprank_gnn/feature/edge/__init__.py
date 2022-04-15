from abc import ABC, abstractmethod
from typing import Dict

import numpy

from deeprank_gnn.models.graph import Edge


class EdgeFeatureProvider(ABC):
    "Each implementation of this class should provide edge features"

    @abstractmethod
    def get_features(self, edge: Edge) -> Dict[str, numpy.ndarray]:
        """ Each implementation of this method should return a dictionary,
            containing the feature value per feature name for this edge.
        """
        pass
