# Use an official Python image as the base image
FROM python:3.9-slim

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

# Install testing dependencies
RUN python -m pip install --upgrade pip \
    && pip install flake8 pre-commit pytest pytest-mock pytest-split pytest-cov types-setuptools

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

# Set up Julia environment (ACEpotentials.jl interface)
RUN julia -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(Pkg.Registry.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry")); Pkg.add("ACEpotentials"); Pkg.add("DataFrames"); Pkg.add("CSV")'

# Set the working directory
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Set the default command to bash
CMD ["bash"]
