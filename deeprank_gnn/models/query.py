from enum import Enum


class QueryType(Enum):
    PROTEIN_PROTEIN_INTERFACE = 1
    SINGLE_RESIDUE_VARIANT = 2


class Query:
    def __init__(self, type_, distance_cutoff, model, target=None):
        self._type = type_
        self._distance_cutoff = distance_cutoff
        self._model = model
        self._target = target

class QueryDataset:
    def __init__(self):
        self._queries = []

    def add(self, query):
        self._queries.append(query)

    @property
    def queries(self):
        return self._queries

    def __contains__(self, query):
        return query in self._queries

