# Installation

## Dependencies

Before installing deeprank2 you need to install some dependencies. We advise to use a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with Python >= 3.9 installed.

* [msms](https://ssbio.readthedocs.io/en/latest/instructions/msms.html): `conda install -c bioconda msms`. *For MacOS with M1 chip users*: you can follow [these instructions](https://ssbio.readthedocs.io/en/latest/instructions/msms.html).
* [PyTorch](https://pytorch.org/):
  * CPU only: `conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch`
  * if using GPU: `conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia`
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html): `conda install pyg -c pyg`
* [Dependencies for pytorch geometric from wheels](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-wheels): `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`.
  - Here, `${TORCH}` and `${CUDA}` should be replaced by the pytorch and CUDA versions installed. You can find these using:
    - `python -c "import torch; print(torch.__version__)"` and
    - `python -c "import torch; print(torch.version.cuda)"`
      - if this returns `None`, use `cpu` instead
  - For example: `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html`
* Check if [DSSP 4](https://swift.cmbi.umcn.nl/gv/dssp/) is installed: `dssp --version`
  * if this gives an error or shows a version lower than 4:
    * on ubuntu 22.04 or newer: `sudo apt-get install dssp`.
      * If the package cannot be located, first run `sudo apt-get update`.
    * on older versions of ubuntu or on mac or lacking sudo priviliges: install from [here](https://github.com/pdb-redo/dssp), following the instructions listed.
* Check if gcc is installed: `gcc --version`.
  * if this gives an error, run `sudo apt-get install gcc`.

* For MacOS with M1 chip (otherwise ignore this): `conda install pytables`

## DeepRank2 Package

Once the dependencies installed, you can install the latest release of deeprank2 using the PyPi package manager:

```bash
pip install deeprank2
```

Alternatively, get all the new developments by cloning the repo and installing the code with

```bash
git clone https://github.com/DeepRank/deeprank2
cd deeprank2
pip install -e ./
```

## Test installation

If you have installed the package from a cloned repository (second option above), you can check that all components were installed correctly, using pytest.
The quick test should be sufficient to ensure that the software works, while the full test (a few minutes) will cover a much broader range of settings to ensure everything is correct.

First, install [pytest](https://docs.pytest.org/): `pip install pytest`.
Then run `pytest tests/test_integration.py` for the quick test or just `pytest` for the full test (expect a few minutes to run).

## Contributing
If you would like to contribute to the package in any way, please see [our guidelines](CONTRIBUTING.rst).
