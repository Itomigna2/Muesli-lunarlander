FROM ubuntu:22.04

RUN apt-get update && apt-get install -y git \
    python3-pip \
    vim

RUN git clone https://github.com/Itomigna2/Muesli-lunarlander.git

WORKDIR /Muesli-lunarlander


