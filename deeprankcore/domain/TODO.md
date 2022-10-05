# My to do list for this issue and #145:

## [ ] Update FEATURENAMEs (done until res_type)
    - [X] make sure to search for strings as well
    - [X] move all this into submodules
    - [X] Double check all the tests for node/edge features
    - [X] Double check README, quickstart, and examples
        - [X] Call features from module instead?
    - [X] Check featurename changes suggested by Giulia
    - [X] Move domain.storage
        - [x] delete domain/storage.py
    - [x] delete domain/feature.py
    - [ ] shorten/clarify some of the feature names?

## [ ] Update clustering groups/names
    - [X] Update cluster naming to one level shallower and prefixing the clustering-type
    - [ ] Test whether this works
        - especially suspect are Trainer.py lines 721/728: `method_grp.create_dataset(groups.DEPTH0, data=cluster.cpu())` (and DEPTH1)

## [ ] Changes dependent on other PRs
    - [ ] Finalize task PR (#200) and atom feature PR (#195) and merge them into this branch; then: 
      - [ ] Make a targettypes module inside domain or inside domain.features for default targets
      - [ ] Move atom feature names into domain/features/nodefeats.py

## [ ] Remake the test *.pdb files
    - I'll need Giulia's help for this

## [ ] Specific feature stuff
    - [ ] Assign currently unused features
        - [x] CHARGE + DIFFCHARGE
        - [ ] CHAINID
            - Not sure whether this is relevant standalone (unlikely) or just as intermediary to SAMECHAIN
            - Don't know how to get this info
                - models/structure has a Chain class
        - [ ] SAMECHAIN always set to 1
        - [X] check whether all other features are actually assigned
            - tools.visualization, line 83: `graph.edges[node1_name, node2_name][edgefeats.SAMECHAIN] = 1.0` without any if statement
            - currently there is no chain info anyway, but once we have that, this would be useful
            - would need to find out how to extract CHAINID from the two nodes of this edge
    - [ ] Remove VARIANTCONSERVATION or add VARIANTXXXXXXX for other features
        - If we make them, then rethink nomenclature
    - [x] Research what happens when onehots are subtracted (feature.aminoacid.py > DIFFPOLARITY)
        - It turns into a 0/1/-1 -type vector, so A-OK  

## [ ] Rethink definitions of deeprankcore/domain vs deeprankcore/models
    - [X] deeprankcore/domain/amino-acid --> add to deeprankcore/models/amino-acid
    - [X] deeprankcore/models/polarity --> add to deeprankcore/models/amino-acid
    - [ ] deeprankcore/models/error --> move to deeprankcore/domain/errors (and start filling it up)
    - [ ] potentially merge the two folders