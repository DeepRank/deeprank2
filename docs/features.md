# Features

Features implemented in the code-base are defined in `deeprankcore.feature`. Users can also add customized features by creating a new module and inserting it in `deeprankcore.feature` subpackage; the only requirement in the latter case is that the module must implement the `add_features` function, that will be used in `deeprankcore.models.query` to build the graph with nodes' and edges' features:
```python
def add_features(
    pdb_path: str, 
    graph: Graph, 
    *args, **kwargs
    ):
    pass
```

The following is a brief description of the features already implemented in the code-base, for each features' module:
- `amino_acid`
- `atom`
- `atomic_contact`
- `biopython`
- `bsa`
- `pssm`
- `sasa`
- `query`
