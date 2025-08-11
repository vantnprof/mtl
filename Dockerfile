# Use official Python image

FROM continuumio/miniconda3


# Set working directory

WORKDIR /work/vantn/mtl/


# Copy requirements if available

COPY environment.yml .


# Install dependencies

RUN conda env create -f environment.yml


# Copy project files

COPY . /work/vantn/mtl/



CMD ["/bin/bash"] 