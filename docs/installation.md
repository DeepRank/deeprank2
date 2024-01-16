# Installation

The package officially supports ubuntu-latest OS only, whose functioning is widely tested through the continuous integration workflows.

## Dependencies

Before installing deeprank2 you need to install some dependencies. We advise to use a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with Python >= 3.10 installed. The following dependency installation instructions are updated as of 14/09/2023, but in case of issues during installation always refer to the official documentation which is linked below:

- [MSMS](https://anaconda.org/bioconda/msms): `conda install -c bioconda msms`.
  - [Here](https://ssbio.readthedocs.io/en/latest/instructions/msms.html) for MacOS with M1 chip users.
- [PyTorch](https://pytorch.org/get-started/locally/)
  - We support torch's CPU library as well as CUDA.
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and its optional dependencies: `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`.
- [DSSP 4](https://swift.cmbi.umcn.nl/gv/dssp/)
- Check if `dssp` is installed: `dssp --version`. If this gives an error or shows a version lower than 4:
- on ubuntu 22.04 or newer: `sudo apt-get install dssp`. If the package cannot be located, first run `sudo apt-get update`.
- on older versions of ubuntu or on mac or lacking sudo priviliges: install from [here](https://github.com/pdb-redo/dssp), following the instructions listed.
- [GCC](https://gcc.gnu.org/install/)
- Check if gcc is installed: `gcc --version`. If this gives an error, run `sudo apt-get install gcc`.
- For MacOS with M1 chip users only install [the conda version of PyTables](https://www.pytables.org/usersguide/installation.html).

## Deeprank2 Package

Once the dependencies are installed, you can install the latest stable release of deeprank2 using the PyPi package manager:

```bash
pip install deeprank2
```

Alternatively, get all the new developments by cloning the repo and installing the editable version of the package with:

```bash
git clone https://github.com/DeepRank/deeprank2
cd deeprank2
pip install -e .'[test]'
```

The `test` extra is optional, and can be used to install test-related dependencies useful during the development.

## Test installation

If you have installed the package from a cloned repository (second option above), you can check that all components were installed correctly, using pytest.
The quick test should be sufficient to ensure that the software works, while the full test (a few minutes) will cover a much broader range of settings to ensure everything is correct.

Run `pytest tests/test_integration.py` for the quick test or just `pytest` for the full test (expect a few minutes to run).

## Contributing

If you would like to contribute to the package in any way, please see [our guidelines](CONTRIBUTING.rst).
