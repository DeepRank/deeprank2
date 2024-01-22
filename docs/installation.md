# Installations

The package officially supports ubuntu-latest OS only, whose functioning is widely tested through the continuous integration workflows.

You can either install DeepRank2 in a [dockerized container](#containerized-installation), which will allow you to run our [tutorial notebooks](https://github.com/DeepRank/deeprank2/tree/main/tutorials), or you can [install the package locally](#localremote-installation).

## Containerized Installation 

In order to try out the package without worrying about your OS and without the need of installing all the required dependencies, we created a `Dockerfile` that can be used for taking care of everything in a suitable container. After having cloned the repository and installed [Docker](https://docs.docker.com/engine/install/), run the following commands (you may need to have sudo permission) from the root of the repository.

Build the Docker image:

```bash
docker build -t deeprank2 .
```

Run the Docker container:

```bash
docker run -p 8888:8888 deeprank2
```

This assumes that your application inside the container is listening on port 8888, and you want to map it to port 8888 on your host machine. Open a browser and go to `http://localhost:8888` to access the application running inside the Docker container and run the tutorials' notebooks.

More details about the tutorials' content can be found [here](https://github.com/DeepRank/deeprank2/blob/main/tutorials/TUTORIAL.md). Note that in the docker container only the raw PDB files are downloaded, needed as a starting point for the tutorials. You can obtain the processed HDF5 files by running the `data_generation_xxx.ipynb` notebooks. Because Docker containers are limited in memory resources, we limit the number of data points processed in the tutorials'. Please install the package locally to fully leverage its capabilities.

After running the tutorials, you may want to remove the (quite large) Docker image from your machine. In this case, remember to [stop the container](https://docs.docker.com/engine/reference/commandline/stop/) and then [remove the image](https://docs.docker.com/engine/reference/commandline/image_rm/). More general information about Docker can be found on the [official website docs](https://docs.docker.com/get-started/).

## Local/remote installation

### Non-pythonic dependencies

Instructions are up to date as of 19 Jan 2024.

Before installing DeepRank2 you need to install some dependencies:

*  [GCC](https://gcc.gnu.org/install/)
    * Check if gcc is installed: `gcc --version`. If this gives an error, run `sudo apt-get install gcc`. 

### Pythonic dependencies

Instructions are up to date as of 19 Jan 2024.

Then, you can use the YML file we provide for creating a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) containing the latest stable release of the package and all the other necessary conda and pip dependencies (CPU only, Python 3.10):

```bash
# Ensure you are in your base environment
conda activate
# Create the environment
conda env create -f env/environment.yml
# Activate the environment
conda activate deeprank2
```

Alternatively, if you are a MacOS user, if the YML file installation is not successfull, or if you want to use CUDA or Python 3.11, you can install each dependency separately, and then the latest stable release of the package using the PyPi package manager. Also in this case, we advise to use a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). In case of issues during installation, please refer to the official documentation for each package (linked below), as our instructions may be out of date:

*  [DSSP 4](https://anaconda.org/sbl/dssp): `conda install -c sbl dssp`.
*  [MSMS](https://anaconda.org/bioconda/msms): `conda install -c bioconda msms`.
    * [Here](https://ssbio.readthedocs.io/en/latest/instructions/msms.html) for MacOS with M1 chip users.
*  [PyTorch](https://pytorch.org/get-started/locally/)
*  [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) `conda install pyg -c pyg`
    * Also install all [optional additions to PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-wheels), namely: `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`.
*  For MacOS with M1 chip users only install [the conda version of PyTables](https://www.pytables.org/usersguide/installation.html).

#### Install DeepRank2

Finally do:

```bash
pip install deeprank2
```

Alternatively, get the latest updates by cloning the repo and installing the editable version of the package with:

```bash
git clone https://github.com/DeepRank/deeprank2
cd deeprank2
pip install -e .'[test]'
```

The `test` extra is optional, and can be used to install test-related dependencies useful during the development.

### Test installation

If you have installed the package from a cloned repository (the latter option above), you can check that all components were installed correctly, using pytest (run `pip install pytest` if you did not install it above).
The quick test should be sufficient to ensure that the software works, while the full test (a few minutes) will cover a much broader range of settings to ensure everything is correct.

Run `pytest tests/test_integration.py` for the quick test or just `pytest` for the full test (expect a few minutes to run).

# Contributing

If you would like to contribute to the package in any way, please see [our guidelines](CONTRIBUTING.rst).

The following section serves as a first guide to start using the package, using protein-protein Interface (PPI) queries as example. For an enhanced learning experience, we provide in-depth [tutorial notebooks](https://github.com/DeepRank/deeprank2/tree/main/tutorials) for generating PPI data, generating SVR data, and for the training pipeline.
For more details, see the [extended documentation](https://deeprank2.rtfd.io/).
