# Use Ubuntu as the base image
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    python3.10 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR bsense_main

RUN git init -q && git remote add origin https://github.com/mtang724/BSENSE-in-cabin.git
RUN git fetch -q origin && git checkout -q baseline
RUN pip install --no-cache-dir -r requirements.txt
