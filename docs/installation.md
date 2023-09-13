# Installation

## Dependencies

Before installing deeprank2 you need to install some dependencies. We advise to use a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with Python >= 3.9 installed. Follow the official documentation linked below:

*  [MSMS](https://ssbio.readthedocs.io/en/latest/instructions/msms.html)
  *  [Here](https://ssbio.readthedocs.io/en/latest/instructions/msms.html) for MacOS with M1 chip users
*  [PyTorch](https://pytorch.org/get-started/locally/)
  *  We support torch's CPU library as well as CUDA
*  [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and its optional dependencies: `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`
*  [DSSP 4](https://swift.cmbi.umcn.nl/gv/dssp/)
*  [GCC](https://gcc.gnu.org/install/)
*  For MacOS with M1 chip users only install [the conda version of PyTables](https://www.pytables.org/usersguide/installation.html)

## Deeprank2 Package

Once the dependencies are installed, you can install the latest stable release of deeprank2 using the PyPi package manager:

```bash
pip install deeprank2
```

Alternatively, get all the new developments by cloning the repo and installing the editable version of the package with

```bash
git clone https://github.com/DeepRank/deeprank2
cd deeprank2
pip install -e .
```

## Test installation

If you have installed the package from a cloned repository (second option above), you can check that all components were installed correctly, using pytest.
The quick test should be sufficient to ensure that the software works, while the full test (a few minutes) will cover a much broader range of settings to ensure everything is correct.

First, install [pytest](https://docs.pytest.org/): `pip install pytest`.
Then run `pytest tests/test_integration.py` for the quick test or just `pytest` for the full test (expect a few minutes to run).

## Contributing
If you would like to contribute to the package in any way, please see [our guidelines](CONTRIBUTING.rst).
