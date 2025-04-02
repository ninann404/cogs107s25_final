FROM jupyter/base-notebook:python-3.11.6

LABEL name="cogniverse-pymc-cli"
LABEL description="PyMC 5.10 CLI environment"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Remove inherited health check
HEALTHCHECK NONE

USER root

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git-core git mercurial subversion \
    build-essential \
    byobu \
    curl \
    htop \
    libcupti-dev \
    libfreetype6-dev \
    libzmq3-dev \
    pkg-config \
    rsync \
    software-properties-common \
    unzip \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up group and user permissions
RUN groupadd -g 1000 jovyan && usermod -g jovyan jovyan
RUN echo "jovyan ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID

# Create conda environment
RUN <<EOF cat > environment.yml
name: pymc
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11.6
  - numpy=1.25.2
  - scipy
  - pandas
  - matplotlib
  - seaborn
  - arviz
  - xarray
  - netcdf4
  - pytables
  - jupyterlab
  - notebook
  - ipywidgets
  - tqdm
  - libblas
  - libcblas
  - liblapack
  - pip
  - pip:
    - pymc==5.10.0
    - jupyter-server-proxy
    - jupyter-resource-usage
EOF

RUN mamba env create -f environment.yml \
    && conda clean --all -f -y \
    && rm environment.yml

RUN mkdir -p /home/jovyan/project
WORKDIR /home/jovyan/project

# For running from bash
SHELL ["/bin/bash","-c"]
RUN echo "conda activate pymc" >> ~/.bashrc && \
    source ~/.bashrc

# Preinstall VSCode server components
RUN mkdir -p /home/jovyan/.vscode-server/extensions \
    /home/jovyan/.vscode-server-insiders/extensions

# Default to bash
CMD ["sleep", "infinity"]
