# Table of contents

- [Table of contents](#table-of-contents)
- [Installation](#installation)
  - [Containerized Installation](#containerized-installation)
  - [Local/remote installation](#localremote-installation)
      - [YML file installation](#yml-file-installation)
    - [Manual installation](#manual-installation)
      - [Testing DeepRank2 installation](#testing-deeprank2-installation)
- [Contributing](#contributing)

# Installation

There are two ways to install DeepRank2:

1. In a [dockerized container](#containerized-installation). This allows you to use DeepRank2, including all the notebooks within the container (a protected virtual space), without worrying about your operating system or installation of dependencies.
   - We recommend this installation for inexperienced users and to learn to use or test our software, e.g. using the provided [tutorials](tutorials/TUTORIAL.md). However, resources might be limited in this installation and we would not recommend using it for large datasets or on high-performance computing facilities.
2. [Local installation](#localremote-installation) on your system. This allows you to use the full potential of DeepRank2, but requires a few additional steps during installation.
   - We recommend this installation for more experienced users, for larger projects, and for (potential) [contributors](#contributing) to the codebase.

## Containerized Installation

In order to try out the package without worrying about your OS and without the need of installing all the required dependencies, we created a `Dockerfile` that can be used for taking care of everything in a suitable container.

For this, you first need to install [Docker](https://docs.docker.com/engine/install/) on your system. Then run the following commands. You may need to have sudo permission for some steps, in which case the commands below can be preceded by `sudo`:

```bash
# Clone the DeepRank2 repository and enter its root directory
git clone https://github.com/DeepRank/deeprank2
cd deeprank2

# Build and run the Docker image
docker build -t deeprank2 .
docker run -p 8888:8888 deeprank2
```

Next, open a browser and go to `http://localhost:8888` to access the application running inside the Docker container. From there you can use DeepRank2, e.g. to run the tutorial notebooks.

More details about the tutorials' contents can be found [here](https://github.com/DeepRank/deeprank2/blob/main/tutorials/TUTORIAL.md). Note that in the docker container only the raw PDB files are downloaded, which needed as a starting point for the tutorials. You can obtain the processed HDF5 files by running the `data_generation_xxx.ipynb` notebooks. Because Docker containers are limited in memory resources, we limit the number of data points processed in the tutorials. Please [install the package locally](#localremote-installation) to fully leverage its capabilities.

If after running the tutorials you want to remove the (quite large) Docker image from your machine, you must first [stop the container](https://docs.docker.com/engine/reference/commandline/stop/) and can then [remove the image](https://docs.docker.com/engine/reference/commandline/image_rm/). More general information about Docker can be found on the [official website docs](https://docs.docker.com/get-started/).

## Local/remote installation

Local installation is formally only supported on the latest stable release of ubuntu, for which widespread automated testing through continuous integration workflows has been set up. However, it is likely that the package runs smoothly on other operating systems as well.

Before installing DeepRank2 please ensure you have [GCC](https://gcc.gnu.org/install/) installed: if running `gcc --version` gives an error, run `sudo apt-get install gcc`.

#### YML file installation

You can use the provided YML file for creating a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) containing the latest stable release of DeepRank2 and all its dependencies.
This will install the CPU-only version of DeepRank2 on Python 3.10.
Note that this will not work for MacOS. Do the [Manual Installation](#manual-installation) instead.

```bash
# Clone the DeepRank2 repository and enter its root directory
git clone https://github.com/DeepRank/deeprank2
cd deeprank2

# Ensure you are in your base environment
conda activate
# Create the environment
conda env create -f env/environment.yml
# Activate the environment
conda activate deeprank2
```

See instructions below to [test](#testing-deeprank2-installation) that the installation was succesful.

### Manual installation

If you want to use the GPUs, choose a specific python version, are a MacOS user, or if the YML installation was not succesful, you can install the package manually. We advise to do this inside a [conda virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
If you have any issues during installation of dependencies, please refer to the official documentation for each package (linked below), as our instructions may be out of date (last tested on 19 Jan 2024):

- [DSSP 4](https://anaconda.org/sbl/dssp): `conda install -c sbl dssp`
- [MSMS](https://anaconda.org/bioconda/msms): `conda install -c bioconda msms`
  - [Here](https://ssbio.readthedocs.io/en/latest/instructions/msms.html) for MacOS with M1 chip users.
- [PyTorch](https://pytorch.org/get-started/locally/): `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
  - Pytorch regularly publishes updates and not all newest versions will work stably with DeepRank2. Currently, the package is tested using [PyTorch 2.1.1](https://pytorch.org/get-started/previous-versions/#v211).
  - We support torch's CPU library as well as CUDA.
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and its optional dependencies: `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`.
  - The exact command to install pyg will depend on the version of pytorch you are using. Please refer to the source's installation instructions (we recommend using the pip installation for this as it also shows the command for the dependencies).
- For MacOS with M1 chip users: install [the conda version of PyTables](https://www.pytables.org/usersguide/installation.html).

Finally install deeprank2 itself: `pip install deeprank2`.

Alternatively, get the latest updates by cloning the repo and installing the editable version of the package with:

```bash
git clone https://github.com/DeepRank/deeprank2
cd deeprank2
pip install -e .'[test]'
```

The `test` extra is optional, and can be used to install test-related dependencies, useful during development.

### Testing DeepRank2 installation

You can check that all components were installed correctly, using pytest. We especially recommend doing this in case you installed DeepRank2 and its dependencies manually (the latter option above).

The quick test should be sufficient to ensure that the software works, while the full test (a few minutes) will cover a much broader range of settings to ensure everything is correct.

First run `pip install pytest`, if you did not install it above. Then run `pytest tests/test_integration.py` for the quick test or just `pytest` for the full test (expect a few minutes to run).

# Contributing

If you would like to contribute to the package in any way, please see [our guidelines](CONTRIBUTING.rst).
