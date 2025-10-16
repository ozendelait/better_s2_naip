FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG UID
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# Install necessary packages and Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    bash \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    git \
    tmux \
    vim \
    curl \
    libjpeg-dev \
    zlib1g-dev \
    tzdata \
    sudo && \
    ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.9 and necessary tools
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-distutils \
    python3.9-venv

# Install Python 3.9 and dev headers
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-distutils \
    python3.9-venv \
    python3.9-dev \
    build-essential \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_VERSION=3.6.4

# Install pip for Python 3.9
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Copy requirements.txt into the image
COPY requirements.txt /workspace/requirements.txt

# Install Python packages from requirements.txt using Python 3.9
RUN python3.9 -m pip install --no-cache-dir -r /workspace/requirements.txt

# User setup (this allows writing to mapped volumes as the host user UID)
RUN apt-get update && apt-get install -y sudo && \
    adduser --disabled-password --gecos "" udocker && \
    adduser udocker sudo && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN usermod -u ${UID} udocker
RUN DOCKER_UID_BUILT=${UID}
USER udocker
RUN sudo chown -R udocker:udocker /workspace

# Set the working directory
WORKDIR /workspace

# Make python3.9 the default python command
RUN echo "alias python='python3.9'" >> ~/.bashrc