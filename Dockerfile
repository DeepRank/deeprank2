# TODO: remove the tests and add the tutorial notebooks
# TODO: download the data maybe, only the pdb files, discuss further

# Pull base image.
FROM --platform=linux/x86_64 continuumio/miniconda3:23.10.0-1

# Add files.
ADD ./tests /home/deeprank2/tests 
ADD ./env/environment.yml /home
ADD ./env/requirements.txt /home

# Install.
RUN \
  apt update -y && \
  ## GCC
  apt install -y gcc && \
  ## DSSP
  wget https://github.com/PDB-REDO/dssp/releases/download/v4.4.0/mkdssp-4.4.0-linux-x64 && \
  mv mkdssp-4.4.0-linux-x64 /usr/local/bin/mkdssp && \
  chmod a+x /usr/local/bin/mkdssp && \
  ## conda and pip deps
  conda env create -f /home/environment.yml

RUN echo "source activate deeprank2" > ~/.bashrc
ENV PATH /opt/conda/envs/deeprank2/bin:$PATH

# Define working directory.
WORKDIR /home/deeprank2

# Define default command.
CMD ["bash"]
# CMD ["jupyter", "lab"]

# not use it if you don't run the notebooks. 
# EXPOSE 9999