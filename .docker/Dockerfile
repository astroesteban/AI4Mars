ARG VARIANT="24.04"
FROM ubuntu:${VARIANT} AS base

# disable prompting during package installation
ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    ssh \
    sshfs \
    openssh-server \
    openssh-client \
    axel \
    g++ \
    make \
    ca-certificates \
    vim \
    bash-completion \
    ffmpeg \
    libsm6 \
    libxext6 \
    nvtop \
    && rm -rf /var/lib/apt/lists/* \
    && echo "source /workspace/.docker/scripts/better_bash_prompt.sh" >> ~/.bashrc

# install miniconda
ARG CONDA_INSTALL_PREFIX="/root/.conda"
ENV CONDA_INSTALL_PREFIX=${CONDA_INSTALL_PREFIX}
ENV PATH="${CONDA_INSTALL_PREFIX}/bin:${PATH}"
ARG PATH="${CONDA_INSTALL_PREFIX}/bin:${PATH}"

# Install Miniconda with Python 3.12
RUN axel -q -n $(nproc) https://repo.anaconda.com/miniconda/Miniconda3-py312_24.3.0-0-Linux-x86_64.sh -o /root/miniconda_install.sh \
  && bash /root/miniconda_install.sh -b -p ${CONDA_INSTALL_PREFIX} \
  && rm /root/miniconda_install.sh \
  && echo "Running $(conda --version)" \
  && conda init bash \
  && . /root/.bashrc \
  && conda install mamba -n base -c conda-forge  -y -q

# setup the conda virtual env
COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml \
  && rm /tmp/environment.yml

ENV PYTHON_INSTALL_LOC=${CONDA_INSTALL_PREFIX}/envs/ai4mars/bin/python

RUN mkdir ~/.kaggle && \
    ln -s /run/secrets/kaggle_json ~/.kaggle/kaggle.json
