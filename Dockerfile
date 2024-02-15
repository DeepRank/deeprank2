# Pull base image
FROM --platform=linux/x86_64 condaforge/miniforge3:23.3.1-1

# Add files
ADD ./tutorials /home/deeprank2/tutorials
ADD ./env/deeprank2-docker.yml /home/deeprank2
ADD ./env/requirements-docker.txt /home/deeprank2

RUN \
  # Install dependencies and package
  apt update -y && \
  apt install unzip -y && \
  ## GCC
  apt install -y gcc && \
  ## Create the environment and install the dependencies
  mamba env create -f /home/deeprank2/deeprank2-docker.yml && \
  ## Activate the environment automatically when entering the container
  echo "source activate deeprank2" >~/.bashrc && \
  # Get the data for running the tutorials
  if [ -d "/home/deeprank2/tutorials/data_raw" ]; then rm -Rf /home/deeprank2/tutorials/data_raw; fi && \
  if [ -d "/home/deeprank2/tutorials/data_processed" ]; then rm -Rf /home/deeprank2/tutorials/data_processed; fi && \
  wget https://zenodo.org/records/8349335/files/data_raw.zip && \
  unzip data_raw.zip -d data_raw && \
  mv data_raw /home/deeprank2/tutorials

ENV PATH /opt/conda/envs/deeprank2/bin:$PATH

# Define working directory
WORKDIR /home/deeprank2

# Define default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--NotebookApp.token=''","--NotebookApp.password=''", "--allow-root"]
