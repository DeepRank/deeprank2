My to do list for this issue and #145:

- [X] Update FEATURENAMEs (done until res_type)
    - [X] make sure to search for strings as well
    - [X] move featurenames into dict instead of individual strings??
    - [X] move all this into submodules
    - [X] Double check all the tests for node/edge features
    - [ ] Double check README, quickstart, and examples
        - [ ] Call features from module instead?
    - [ ] Check featurename changes suggested by Giulia
    - [X] Move domain.storage
        - [ ] delete domain/storage.py
    - [ ] delete domain/feature.py

- [ ] Fix clustering groups/names

- [ ] Make a targettypes module inside domain or inside domain.features for default targets
    This could easily conflict with the task-PR (#200) 

- [ ] Remake the test *.pdb files
    I'll need Giulia's help for this

- [ ] Specific feature stuff
    - [ ] Assign currently unused features
        - [ ] CHAIN
        - [ ] CHARGE + DIFFCHARGE
        - [ ] CONSERVATION
        - [ ] CONSERVATIONDIFFERENCE
        - [ ] others?
    - [ ] INTERFACE always set to 1
        tools.visualization, line 83: `graph.edges[node1_name, node2_name][edgefeats.INTERFACE] = 1.0`
        check what to do about this; probably do a check for which chain it's on
    - [ ] Remove VARIANTCONSERVATION or add VARIANTXXXXXXX for other features
        - [ ] If make them, then rethink nomenclature
    - [ ] Research what happens when onehots are subtracted (feature.aminoacid.py > DIFFPOLARITY)



