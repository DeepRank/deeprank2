# Pull base image
FROM --platform=linux/amd64 ubuntu:22.04

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=24.3.0-0
ARG TARGETPLATFORM

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN \
  ## Install apt dependencies
  apt-get update && \
  apt-get install --no-install-recommends --yes \
      wget bzip2 unzip ca-certificates \
      git && \
  ## Download and install Miniforge
  wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-x86_64.sh -O /tmp/miniforge.sh && \
  /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
  rm /tmp/miniforge.sh && \
  echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
  echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

ADD ./env/deeprank2.yml /home/deeprank2

RUN \
  ## Create the environment and install the dependencies
  mamba env create -f /home/deeprank2/deeprank2.yml && \
  conda install -n deeprank2 conda-forge::gcc && \
  ## Activate the environment and install pip packages
  conda run -n deeprank2 pip install deeprank2 && \
  ## Activate the environment automatically when entering the container
  echo "source activate deeprank2" >~/.bashrc && \
  # Get the data for running the tutorials
  if [ -d "/home/deeprank2/tutorials/data_raw" ]; then rm -Rf /home/deeprank2/tutorials/data_raw; fi && \
  if [ -d "/home/deeprank2/tutorials/data_processed" ]; then rm -Rf /home/deeprank2/tutorials/data_processed; fi && \
  wget https://zenodo.org/records/8349335/files/data_raw.zip && \
  unzip data_raw.zip -d data_raw && \
  mv data_raw /home/deeprank2/tutorials && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  conda clean --tarballs --index-cache --packages --yes && \
  find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
  find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
  conda clean --force-pkgs-dirs --all --yes

ADD ./tutorials /home/deeprank2/tutorials

ENV PATH /opt/conda/envs/deeprank2/bin:$PATH

# Define working directory
WORKDIR /home/deeprank2

# Define default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--NotebookApp.token=''","--NotebookApp.password=''", "--allow-root"]
