My to do list for this issue and #145:

- [X] Update FEATURENAMEs (done until res_type)
    - [X] make sure to search for strings as well
    - [X] move featurenames into dict instead of individual strings??
    - [X] move all this into submodules
    - [X] Double check all the tests for node/edge features
    - [X] Double check README, quickstart, and examples
        - [X] Call features from module instead?
    - [X] Check featurename changes suggested by Giulia
    - [X] Move domain.storage
        - [x] delete domain/storage.py
    - [x] delete domain/feature.py

- [ ] Fix clustering groups/names

- [ ] Make a targettypes module inside domain or inside domain.features for default targets
    This could easily conflict with the task-PR (#200), so maybe finalize that first, then merge main into here before doing this

- [ ] Remake the test *.pdb files
    I'll need Giulia's help for this

- [ ] Specific feature stuff
    - [ ] Assign currently unused features
        - [x] CHARGE + DIFFCHARGE
        - [ ] CHAIN
            - Not sure how to extract this info, but it seems relevant
        - [ ] SAMECHAIN always set to 1
            tools.visualization, line 83: `graph.edges[node1_name, node2_name][edgefeats.SAMECHAIN] = 1.0` without any if statement
            currently there is no chain info anyway, but once we have that, this would be useful
            check what to do about this; probably do a check for which chain it's on
        - [ ] others?
    - [ ] Remove VARIANTCONSERVATION or add VARIANTXXXXXXX for other features
        - [ ] If make them, then rethink nomenclature
    - [ ] Research what happens when onehots are subtracted (feature.aminoacid.py > DIFFPOLARITY)

- [ ] Move atom feature names in
    Need to merge atom feature PR first (#195)

