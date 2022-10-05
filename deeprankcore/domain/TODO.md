My to do list for this issue and #145:

- [X] Update FEATURENAMEs (done until res_type)
    - [X] make sure to search for strings as well
    - [X] move featurenames into dict instead of individual strings??
    - [X] move all this into submodules
    - [ ] delete domain/feature.py

- [ ] Fix clustering groups/names

- [X] Move domain.storage
    - [ ] delete domain/storage.py

- [ ] Check featurename changes suggested by Giulia

- [ ] Make a targettypes module inside domain or inside domain.features

- [ ] Check README, quickstart, and examples
    - [ ] Call features from module instead?

- [ ] Research what happens when onehots are subtracted (feature.aminoacid.py > DIFFPOLARITY)

- [X] Double check all the tests for node/edge features

- [ ] Remake the test *.pdb files
    I'll need Giulia's help for this

- [ ] Specific feature stuff
    - [ ] Assign currently unused features
        - [ ] CHAIN
        - [ ] CHARGE + DIFFCHARGE
        - [ ] CONSERVATION
        - [ ] CONSERVATIONDIFFERENCE
        - [ ] check the others
    - [ ] INTERFACE always set to 1
        tools.visualization, line 83: `graph.edges[node1_name, node2_name][edgefeats.INTERFACE] = 1.0`
        check what to do about this 
    - [ ] Remove VARIANTCONSERVATION or add VARIANT[XXX] for other features


