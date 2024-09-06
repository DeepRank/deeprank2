# Docking Scores

The following scores have been developed for evaluating the quality of the protein-protein models produced by computational methods (docking models), and all of them compare the structural similarity between the decoys (computationally generated structures) and the experimentally solved native structures. To calculate these measures, the interface between the two interacting protein molecules is defined as any pair of heavy atoms from the two molecules within 5Å of each other.

- `lmrsd` (ligand root mean square deviation) is a float value calculated for the backbone of the shorter chain (ligand) of the model after superposition of the longer chain (receptor). Lower scores represent better matching than higher scores.
- `imrsd` (interface rmsd) is a float value calculated for the backbone atoms of the interface residues (atomic contact cutoff of 10Å) after superposition of their equivalents in the predicted complex (model) Lower scores represent better matching than higher scores.
- `fnat` (fraction of native contacts) is the fraction of native interfacial contacts preserved in the interface of the predicted complex. The score is a float in the range [0, 1], where higher values respresent higher quality.
- `dockq` (docking model quality) is a continuous quality measure for docking models that instead of classifying into different quality groups. It combines fnat, lmrs, and irms and yields a float score in the range [0, 1], where higher values respresent higher quality.
- `binary` (bool): True if the irmsd is lower than 4.0, meaning that the decoy is considered high quality docking model, otherwise False.
- `capri_class` (int). It refers to Critical Assessment of PRedicted Interactions (CAPRI) classification, in which the possible values are: 1 (high quality, irmsd < 1.0), 2 (medium, irmsd < 2.0), 3 (acceptable, irms < 4.0), 4 (incorrect, irmsd >= 4.0)

See https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.10393 for more details about `capri_class`, `lrmsd`, `irmsd`, and `fnat`. See https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879 for more details about `dockq`.

## Compute and Add Docking Scores

The following code snippet shows an example of how to use deeprank2 to compute the docking scores for a given docking model, and how to add one of the scores (e.g., `dockq`) as a target to the already processed data.

```python
from deeprank2.tools.target import add_target, compute_ppi_scores

docking_models = [
    "<path_to_docking_model1.pdb>",
    "<path_to_docking_model2.pdb>"
    ]
ref_models = [
    "<path_to_ref_model1.pdb>",
    "<path_to_ref_model2.pdb>"
]

target_list = ""
for idx, _ in enumerate(docking_models):
    scores = compute_ppi_scores(
        docking_models[idx],
        ref_models[idx])
    dockq = scores['dockq']
    target_list += f"query_id_model{idx} {dockq}\n"

with open("<path_to_target_list.lst>", "w", encoding="utf-8") as f:
    f.write(target_list)

add_target("<path_to_hdf5_file.hdf5>", "dockq", "<path_to_target_list.lst>")

```

After having run the above code snipped, each processed data point within the indicated HDF5 file will contain a new Dataset called "dockq", containing the value computed through `compute_ppi_scores`.
