# Change Log

## 2.0.0

### Main changes

#### Refactor
* refactor: make `preprocess` use all available feature modules as default by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/247
* refactor: move preprocess function to `QueryDataset` class and rename by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/252
* refactor: save preprocessed data into one .hdf5 file as default by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/250
* refactor: clean up `GraphDataset` and `Trainer` class by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/255
* refactor: reorganize deeprankcore.utils.metrics module by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/262
* refactor: fix `transform_sigmoid` logic and move it to `GraphDataset` class by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/288
* refactor: add grid dataset class and make the trainer class work with it. by @cbaakman in https://github.com/DeepRank/deeprank-core/pull/294
* refactor: update deprecated dataloader import by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/310
* refactor: move tests/_utils.py to tests/__init__.py by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/322
* refactor: delete all outputs from unit tests after run by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/324
* refactor: test_contact.py function naming and output by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/372
* refactor: split test contact.py by @joyceljy in https://github.com/DeepRank/deeprank-core/pull/369
* refactor: change __repr__ of AminoAcid to 3 letter code by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/384
* refactor: make feature modules and tests uniform and ditch duplicate code by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/400

#### Features
* feat: improve amino acid features by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/272
* feat: add `test_size` equivalent of `val_size` to Trainer class by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/291
* feat: add the option to have a grid box of different x,y and z dimensions by @cbaakman in https://github.com/DeepRank/deeprank-core/pull/292
* feat: add early stopping to `Trainer.train` by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/303
* feat: add hist module for plotting raw hdf5 files features distributions by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/261
* feat: allow for different loss functions other than the default by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/313
* feat: center the grids as in the old deeprank by @cbaakman in https://github.com/DeepRank/deeprank-core/pull/323
* feat: add data augmentation for grids by @cbaakman in https://github.com/DeepRank/deeprank-core/pull/336
* feat: insert features standardization option in`DeeprankDataset` children classes by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/326
* feat: add log transformation option for plotting features' hist by @joyceljy in https://github.com/DeepRank/deeprank-core/pull/389
* feat: add inter-residue contact (IRC) node features by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/333
* feat: add feature module for secondary structure by @DTRademaker in https://github.com/DeepRank/deeprank-core/pull/387
* feat: use dictionary for flexibly transforming and standardizing features by @joyceljy in https://github.com/DeepRank/deeprank-core/pull/418

#### Fix
* fix: list all submodules imported from deeprankcore.features using pkgutil by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/263
* fix: let `classes` argument be also categorical by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/286
* fix: makes sure that the `map_feature` function can handle single value features. by @cbaakman in https://github.com/DeepRank/deeprank-core/pull/289
* fix: raise exception for invalid optimizer by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/307
* fix: `num_workers` parameter of Dataloader object by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/319
* fix: gpu usage by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/334
* fix: gpu and `entry_names` usage by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/335
* fix: data generation threading locked by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/330
* fix: `__hash__` circular dependency issue by @cbaakman in https://github.com/DeepRank/deeprank-core/pull/341
* fix: make sure that Grid data also has target values, like graph data by @cbaakman in https://github.com/DeepRank/deeprank-core/pull/347
* fix: change the internal structure of the grid data to match the graph data by @cbaakman in https://github.com/DeepRank/deeprank-core/pull/352
* fix: conflicts in package by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/386
* fix: correct usage of nonbond energy for close contacts by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/368
* fix: Incorrect number of datapoints loaded to model by @joyceljy in https://github.com/DeepRank/deeprank-core/pull/397
* fix: pytorch 2.0 by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/406
* fix: covalent bonds cannot link nodes on separate branches by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/408
* fix: `Trainer` error when only `dataset_test` and `pretrained_model` are used by @ntxxt in https://github.com/DeepRank/deeprank-core/pull/413
* fix: check PSSMs by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/401
* fix: only check pssms if conservation module was used by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/425
* fix: epoch number in `test()` and test on the correct model by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/427
* fix: convert list of arrays into arrays before converting to Pytorch tensor by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/438

#### Docs
* docs: add verbose arg to QueryCollection class by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/267
* docs: improve `clustering_method` description and default value by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/293
* docs: uniform docstrings format in modules by @joyceljy
* docs: incorrect usage of Union in Optional type hints by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/370
* docs: improve docs for default exporter and results visualization by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/414
* docs: update feature documentations by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/419
* docs: add instructions for `GridDataset` by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/421
* docs: fix getstarted hierarchy by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/422
* docs: update dssp 4 install instructions by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/437
* docs: change `external_distance_cutoff` and `interface_distance_cutoff` to `distance_cutoff` by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/246

#### Performances
* perf: features.contact by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/220
* perf: suppress warnings in pytest and from PDBParser by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/249
* perf: add try except clause to `_preprocess_one_query` method of `QueryCollection` class by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/264
* perf: improve `process` speed for residue based graph building by @cbaakman in https://github.com/DeepRank/deeprank-core/pull/274
* perf: add `cuda` and `ngpu` parameters to the `Trainer` class by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/311
* perf: accelerate indexing of HDF5 files by @joyceljy in https://github.com/DeepRank/deeprank-core/pull/362

#### Style
* style: restructure deeprankcore package and subpackages by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/240
* style: reorganize features/contact.py by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/260
* style: add .vscode settings.json by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/404

#### Test
* test: make sure that the grid orientation is as in the original deeprank for `ProteinProteinInterfaceAtomicQuery` by @cbaakman in https://github.com/DeepRank/deeprank-core/pull/312
* test: check that the grid for residue-based protein-protein interfaces has the same center and orientation as in the original deeprank. by @cbaakman in https://github.com/DeepRank/deeprank-core/pull/339
* test: improve `utils/test_graph.py` module by @gcroci2 in https://github.com/DeepRank/deeprank-core/pull/420

#### CI
* ci: do not close stale issues or PRs by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/327
* ci: remove incorrect message for stale branches by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/415
* ci: automatically check markdown links by @DaniBodor in https://github.com/DeepRank/deeprank-core/pull/433

### New Contributors
* @joyceljy made their first contribution in https://github.com/DeepRank/deeprank-core/pull/361
* @ntxxt made their first contribution in https://github.com/DeepRank/deeprank-core/pull/413

**Full Changelog**: https://github.com/DeepRank/deeprank-core/compare/v1.0.0...v2.0.0

## 1.0.0

Released on Oct 24, 2022

### Added

* `weight_decay` parameter to NeuralNet #155
* Exporter for generating a unique .csv file containing results per epoch #151
* Automatized testing of all available features modules #163
* `optimizer` parameter to NeuralNet #154
* `atom` node feature #168

### Changed

* `index` parameter of NeuralNet is now called `subset` #159
* `percent` parameter of NeuralNet is now called `val_size`, and the logic behing it has been improved #183
* Aligned the package to PyTorch high-level frameworks #172
  * NeuralNet is now called Trainer 
* Clearer features names #145
* Changed definitions in storage.py #150
* `MAX_COVALENT_DISTANCE` is now 2.1 instead of 3 #205

### Removed

* `threshold` input parameter from NeuralNet #157

## 0.2.0

Released on Aug 10, 2022

### Added

* Automatic version bumping using `bump2version` with `.bumpversion.cfg` #126
* `cffconvert.yml` to the CI workflow #139
* Integration test for the Machine Learning pipeline #95
* The package now is tested also on Python 3.10 #165

### Changed

* Test PyPI package before publishing, by triggering a `workflow_dispatch` event from the Actions tab on `release.yml` workflow file #123
* Coveralls is now working again #124
* Wrong Zenodo entry has been corrected #138
* Improved CUDA support (added for data tensors) #132

## 0.1.1

Released on June 28, 2022

### Added

* Graph class #48
* Tensorboard #15
* CI Linting #30
* Name, affiliation and orcid to `.zenodo.json` #18
* Metrics class #17
* QueryDataset class #53
* Unit tests for NeuralNet class #86
* Error message if you pick the wrong metrics #110
* Unit tests for HDF5Dataset class parameters #82
* Installation from PyPI in the readme #122

### Changed

* `test_process()` does not fail anymore #47
* Tests have been speded up #36
* `multiprocessing.Queue` has been replaced with `multiprocessing.pool.map` in PreProcessor #56
* `test_preprocess.py` does not fail anymore on Mac M1 #74
* It's now possible to pass your own train/test split to NeuralNet class #81
* HDF5Dataset class now is used in the UX #83
* IndexError running `NeuralNet.train()` has been fixed #89
* pip installation has been fixed
* Repository has been renamed deeprank-core, and the package deeprankcore #101
* The zero-division like error from TensorboardBinaryClassificationExporter has been fixed #112
* h5xplorer is installed through `setup.cfg` file #121
* Sphinx docs have been fixed #108
