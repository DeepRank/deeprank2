# Change Log

## Unreleased

### Added

### Changed

### Removed

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
* Unit tests for HDF5DataSet class parameters #82
* Installation from PyPI in the readme #122

### Changed

* `test_process()` does not fail anymore #47
* Tests have been speded up #36
* `multiprocessing.Queue` has been replaced with `multiprocessing.pool.map` in PreProcessor #56
* `test_preprocess.py` does not fail anymore on Mac M1 #74
* It's now possible to pass your own train/test split to NeuralNet class #81
* HDF5DataSet class now is used in the UX #83
* IndexError running `NeuralNet.train()` has been fixed #89
* pip installation has been fixed
* Repository has been renamed deeprank-core, and the package deeprankcore #101
* The zero-division like error from TensorboardBinaryClassificationExporter has been fixed #112
* h5xplorer is installed through `setup.cfg` file #121
* Sphinx docs have been fixed #108