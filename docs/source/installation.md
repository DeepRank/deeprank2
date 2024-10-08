# Installation

There are two ways to install DeepRank2:

1. In a [dockerized container](containerized-installation). This allows you to use DeepRank2, including all the notebooks within the container (a protected virtual space), without worrying about your operating system or installation of dependencies.
   - We recommend this installation for inexperienced users and to learn to use or test our software, e.g. using the provided [tutorials](https://github.com/DeepRank/deeprank2/blob/main/tutorials/TUTORIAL.md). However, resources might be limited in this installation and we would not recommend using it for large datasets or on high-performance computing facilities.
2. [Local installation](#localremote-installation) on your system. This allows you to use the full potential of DeepRank2, but requires a few additional steps during installation.
   - We recommend this installation for more experienced users, for larger projects, and for (potential) [contributors](contributing_link.md) to the codebase.

## Containerized Installation

(containerized-installation)=

We provide a pre-built Docker image hosted on GitHub Packages, allowing you to use DeepRank2 without worrying about installing dependencies or configuring your system. This is the recommended method for trying out the package quickly.

### Pull and Run the Pre-build Docker Image (Recommended)

- Install [Docker](https://docs.docker.com/engine/install/) on your system, if not already installed.
- Pull the latest Docker image from GitHub Packages by running the following command:

```bash
docker pull ghcr.io/deeprank/deeprank2:latest
```

- Run the container from the pulled image:

```bash
docker run -p 8888:8888 ghcr.io/deeprank/deeprank2:latest
```

- Once the container is running, open your browser and navigate to `http://localhost:8888` to access the DeepRank2 application.

From here, you can use DeepRank2, including running the tutorial notebooks. More details about the tutorials can be found [here](https://github.com/DeepRank/deeprank2/blob/main/tutorials/TUTORIAL.md). Note that the Docker container downloads only the raw PDB files required for the tutorials. To generate processed HDF5 files, you will need to run the `data_generation_xxx.ipynb` notebooks. Since Docker containers may have limited memory resources, we reduce the number of data points processed in the tutorials. To fully utilize the package, consider [installing it locally](#localremote-installation).

### Build the Docker Image Manually

If you prefer to build the Docker image yourself or run into issues with the pre-built image, you can manually build and run the container as follows:

- Install [Docker](https://docs.docker.com/engine/install/) on your system, if not already installed.
- Clone the DeepRank2 repository and navigate to its root directory:

```bash
git clone https://github.com/DeepRank/deeprank2
cd deeprank2
```

- Build and run the Docker image:

```bash
docker build -t deeprank2 .
docker run -p 8888:8888 deeprank2
```

- Once the container is running, open your browser and navigate to `http://localhost:8888` to access the DeepRank2 application.

### Removing the Docker Image

If you no longer need the Docker image (which can be quite large), you can remove it after stopping the container. Follow the [container stop](https://docs.docker.com/engine/reference/commandline/stop/) and [remove the image](https://docs.docker.com/engine/reference/commandline/image_rm/) instructions. For more general information on Docker, refer to the [Docker documentation](https://docs.docker.com/get-started/) directly.

## Local/remote Installation

(localremote-installation)=

Local installation is formally only supported on the latest stable release of ubuntu, for which widespread automated testing through continuous integration workflows has been set up. However, it is likely that the package runs smoothly on other operating systems as well.

Before installing DeepRank2 please ensure you have [GCC](https://gcc.gnu.org/install/) installed: if running `gcc --version` gives an error, run `sudo apt-get install gcc`.

## YML File Installation (Recommended)

You can use the provided YML file for creating a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) via [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), containing the latest stable release of DeepRank2 and all its dependencies.
This will install the CPU-only version of DeepRank2 on Python 3.10.
Note that this will not work for MacOS. Do the [manual installation](#manual-installation) instead.

```bash
# Create the environment
mamba env create -f https://raw.githubusercontent.com/DeepRank/deeprank2/main/env/deeprank2.yml
# Activate the environment
conda activate deeprank2
# Install the latest deeprank2 release
pip install deeprank2
```

We also provide a frozen environment YML file located at `env/deeprank2_frozen.yml` with all dependencies set to fixed versions. The `env/deeprank2_frozen.yml` file provides a frozen environment with all dependencies set to fixed versions. This ensures reproducibility of experiments and results by preventing changes in package versions that could occur due to updates or modifications in the default `env/deeprank2.yml`. Use this frozen environment file for a stable and consistent setup, particularly if you encounter issues with the default environment file.

## Manual Installation (Customizable)

(manual-installation)=

If you want to use the GPUs, choose a specific python version (note that at the moment we support python 3.10 only), are a MacOS user, or if the YML installation was not successful, you can install the package manually. We advise to do this inside a [conda virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

You can first create a copy of the `deeprank2.yml` file, place it in your current directory, and remove the packages that cannot be installed properly, or the ones that you want to install differently (e.g., pytorch-related packages if you wish to install the CUDA version), and then proceed with the environment creation by using the edited YML file: `conda env create -f deeprank2.yml` or `mamba env create -f deeprank2.yml`, if you have [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) installed. Then activate the environment, and proceed with installing the missing packages, which might fall into the following list. If you have any issues during installation of dependencies, please refer to the official documentation for each package (linked below), as our instructions may be out of date (last tested on 19 Feb 2024):

- [MSMS](https://anaconda.org/bioconda/msms): [Here](https://ssbio.readthedocs.io/en/latest/instructions/msms.html) for MacOS with M1 chip users.
- [PyTorch](https://pytorch.org/get-started/locally/)
  - Pytorch regularly publishes updates and not all newest versions will work stably with DeepRank2. Currently, the package is tested on ubuntu using [PyTorch 2.1.1](https://pytorch.org/get-started/previous-versions/#v211).
  - We support torch's CPU library as well as CUDA.
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and its optional dependencies: `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`.
  - The exact command to install pyg will depend on the version of pytorch you are using. Please refer to the source's installation instructions (we recommend using the pip installation for this as it also shows the command for the dependencies).
- [FreeSASA](https://freesasa.github.io/python/).

Finally install deeprank2 itself: `pip install deeprank2`.

Alternatively, get the latest updates by cloning the repo and installing the editable version of the package with:

```bash
git clone https://github.com/DeepRank/deeprank2
cd deeprank2
pip install -e .'[test]'
```

The `test` extra is optional, and can be used to install test-related dependencies, useful during development.

## Testing DeepRank2 Installation

If you have cloned the repository, you can check that all components were installed correctly using `pytest`. We especially recommend doing this in case you installed DeepRank2 and its dependencies manually (the latter option above).

The quick test should be sufficient to ensure that the software works, while the full test (a few minutes) will cover a much broader range of settings to ensure everything is correct.

First run `pip install pytest`, if you did not install it above. Then run `pytest tests/test_integration.py` for the quick test or just `pytest` for the full test (expect a few minutes to run).
