# Use an official micromamba image as the base image
ARG PYTHON_VERSION=3.10

FROM mambaorg/micromamba:1.5.10


# Set environment variables for micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ROOT_PREFIX=/opt/conda

# Switch to root to install all dependencies (using non-root user causes permission issues)
USER root

# Make arg accessible to the rest of the Dockerfile
ARG PYTHON_VERSION

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    bc \
    unzip \
    wget \
    gfortran \
    liblapack-dev \
    libblas-dev \
    cmake \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Julia
RUN curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.2-linux-x86_64.tar.gz | tar -xz -C /opt \
    && ln -s /opt/julia-1.9.2/bin/julia /usr/local/bin/julia

# Set up Julia environment (ACEpotentials.jl interface)
RUN julia -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(Pkg.Registry.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry")); Pkg.add("ACEpotentials"); Pkg.add("DataFrames"); Pkg.add("CSV")'

# Install Buildcell

# Define the target directory to download and install AIRSS
WORKDIR /opt/

RUN curl -O https://www.mtg.msm.cam.ac.uk/files/airss-0.9.3.tgz \
    && tar -xf  airss-0.9.3.tgz \
    && rm  airss-0.9.3.tgz \
    && cd airss \
    && make \
    && make install \
    && make neat

# Add Buildcell to PATH
ENV PATH="/opt/airss/bin"

RUN micromamba install -y -n base -c conda-forge \ python=${PYTHON_VERSION} && \
    micromamba clean --all --yes


# Install testing dependencies
RUN python -m pip install --upgrade pip \
    && pip install uv \
    && uv pip install flake8 pre-commit pytest pytest-mock pytest-split pytest-cov types-setuptools
