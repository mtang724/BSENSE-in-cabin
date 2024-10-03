# Use Ubuntu as the base image
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    unzip \
    git \
    python3.10 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y install git-lfs

# Set the working directory in the container
WORKDIR bsense_main
COPY test_data.zip /tmp/test_data.zip

RUN git init -q && git remote add origin https://github.com/mtang724/BSENSE-in-cabin.git
RUN git fetch -q origin && git checkout -q main
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir BSENSE/model_training/test_data \
    && unzip /tmp/test_data.zip -d BSENSE/model_training/
RUN mkdir BSENSE/data_preprocessing/processed_dataset

