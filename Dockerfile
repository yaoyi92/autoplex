# Use an official micromamba image as the base image
ARG PYTHON_VERSION=3.10
FROM mambaorg/micromamba:1.5.10

# Set environment variables for micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ROOT_PREFIX=/opt/conda

# Switch to root to install system dependencies
USER root

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
#RUN curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.2-linux-x86_64.tar.gz | tar -xz -C /opt \
#    && ln -s /opt/julia-1.9.2/bin/julia /usr/local/bin/julia

# Set up Julia environment (ACEpotentials.jl interface)
#RUN julia -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(Pkg.Registry.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry")); Pkg.add("ACEpotentials"); Pkg.add("DataFrames"); Pkg.add("CSV")'

# Install Buildcell
RUN curl -O https://www.mtg.msm.cam.ac.uk/files/airss-0.9.3.tgz \
    && tar -xf airss-0.9.3.tgz \
    && rm airss-0.9.3.tgz \
    && cd airss \
    && make \
    && make install \
    && make neat

# Add Buildcell to PATH
ENV PATH="${PATH}:/airss/bin"

# Switch back to the micromamba default user (typically 'mambauser' user)
USER $MAMBA_USER

RUN micromamba install -y -n base python=${PYTHON_VERSION} && \
    micromamba clean --all --yes

# Create a new environment named "autoplex_test" with the specified Python version
#RUN /opt/conda/bin/micromamba create -n autoplex_test python=${PYTHON_VERSION} --yes

# Activate the autoplex_test environment
#SHELL ["micromamba", "activate", "autoplex_test"]

# Install testing dependencies
RUN python -m pip install --upgrade pip \
    && pip install uv \
    && uv pip install flake8 pre-commit pytest pytest-mock pytest-split pytest-cov types-setuptools

# Set the working directory
# WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
# COPY . /workspace

# Set the default command to bash
# CMD ["bash"]
