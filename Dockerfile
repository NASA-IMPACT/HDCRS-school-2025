# Part of the implementation of this container is based on the Amazon SageMaker Apache MXNet container.
# https://github.com/aws/sagemaker-mxnet-container

FROM nvidia/cuda:12.9.0-base-ubuntu22.04

LABEL maintainer="NASA IMPACT"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.11-dev  && \
    apt-get install -y libgl1  && \
    apt-get install -y python3-pip

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code/

RUN python3.11 -m pip install -U pip

COPY requirements.txt /opt/ml/code/requirements.txt

RUN pip3 install -r /opt/ml/code/requirements.txt --ignore-installed

ENV CUDA_VISIBLE_DEVICES=0,1,2

ENV CUDA_HOME=/usr/local/cuda

ENV FORCE_CUDA="1"

RUN mkdir /opt/ml/code/models

# Copies code under /opt/ml/code where sagemaker-containers expects to find the script to run
COPY code/lib/train.py /opt/ml/code/train.py

COPY code/lib/utils.py /opt/ml/code/utils.py

# Defines train.py as script entry point
ENV SAGEMAKER_PROGRAM /opt/ml/code/train.py
