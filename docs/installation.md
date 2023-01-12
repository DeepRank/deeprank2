# Installation

## Dependencies

Before installing deeprankcore you need to install:

 * [reduce](https://github.com/rlabduke/reduce): follow the instructions in the README of the reduce repository.
    * **How to build it without sudo privileges on a Linux machine**. After having run `make` in the reduce/ root directory, go to reduce/reduce_src/Makefile and modify `/usr/local/` to a folder in your home directory, such as `/home/user_name/apps`. Note that such a folder needs to be added to the PATH in the `.bashrc` file. Then run `make install` from reduce/. 
 * [msms](https://ssbio.readthedocs.io/en/latest/instructions/msms.html): `conda install -c bioconda msms`. *For MacOS with M1 chip users*: you can follow [these instructions](https://ssbio.readthedocs.io/en/latest/instructions/msms.html).
 * [pytorch](https://pytorch.org/get-started/locally/): `conda install pytorch torchvision torchaudio cpuonly -c pytorch` or `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`, for taking advantage of GPUs.
 * [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html): `conda install pyg -c pyg`
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
