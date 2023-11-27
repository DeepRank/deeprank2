# Pull base image
FROM --platform=linux/x86_64 continuumio/miniconda3:23.10.0-1

# Add files
ADD ./tutorials /home/deeprank2/tutorials 
ADD ./env/environment.yml /home
ADD ./env/requirements.txt /home

# Install
RUN \
  apt update -y && \
  apt install unzip -y && \
  ## GCC
  apt install -y gcc && \
  ## DSSP
  wget https://github.com/PDB-REDO/dssp/releases/download/v4.4.0/mkdssp-4.4.0-linux-x64 && \
  mv mkdssp-4.4.0-linux-x64 /usr/local/bin/mkdssp && \
  chmod a+x /usr/local/bin/mkdssp && \
  ## Conda and pip deps
  conda env create -f /home/environment.yml && \
  ## Get the data for running the tutorials
  wget https://zenodo.org/records/8349335/files/data_raw.zip && \
  unzip data_raw.zip -d data_raw && \
  mv data_raw /home/deeprank2/tutorials

# Activate the environment
RUN echo "source activate deeprank2" > ~/.bashrc
ENV PATH /opt/conda/envs/deeprank2/bin:$PATH

# Define working directory
WORKDIR /home/deeprank2

# Define default command
CMD ["bash"]
