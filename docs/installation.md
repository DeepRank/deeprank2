# Installation

## Dependencies

Before installing deeprankcore you need to install:

 * [reduce](https://github.com/rlabduke/reduce): follow the instructions in the README of the reduce repository.
    * **How to build it without sudo privileges on a Linux machine**. After having run `make` in the reduce/ root directory, go to reduce/reduce_src/Makefile and modify `/usr/local/` to a folder in your home directory, such as `/home/user_name/apps`. Note that such a folder needs to be added to the PATH in the `.bashrc` file. Then run `make install` from reduce/. 
 * [msms](https://ssbio.readthedocs.io/en/latest/instructions/msms.html): `conda install -c bioconda msms`. *For MacOS with M1 chip users*: you can follow [these instructions](https://ssbio.readthedocs.io/en/latest/instructions/msms.html).
 * [dssp](https://swift.cmbi.umcn.nl/gv/dssp/): `sudo apt-get install dssp`
    * See [DSSP docs](https://ssbio.readthedocs.io/en/latest/instructions/dssp.html) for installing it on Mac OSX
 * [pytorch](https://pytorch.org/get-started/locally/): 
   * CPU only: `conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch`
   * if using GPU: `conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia`
 * [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html): `conda install pyg -c pyg`
 * [Dependencies for pytorch geometric from wheels](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-wheels): `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`. 
    - Here, `${TORCH}` and `${CUDA}` should be replaced by the pytorch and CUDA versions installed. You can find these using:
      - `python -c "import torch; print(torch.__version__)"` and
      - `python -c "import torch; print(torch.version.cuda)"`
    - For example: `https://data.pyg.org/whl/torch-2.0.0+cpu.html`
 * Only if you have a MacOS with M1 chip, additional steps are needed:
    * `conda install pytables`
    * See [this](https://stackoverflow.com/questions/30145751/python3-cant-find-and-import-pyqt5) solution to install PyQt5 or run `conda install pyqt`

## Deeprank-Core Package

Once the dependencies installed, you can install the latest release of deeprankcore using the PyPi package manager:

```
pip install deeprankcore
```

You can get all the new developments by cloning the repo and installing the code with

```
git clone https://github.com/DeepRank/deeprank-core
cd deeprank-core
pip install -e ./
```
