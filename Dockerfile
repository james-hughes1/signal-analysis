FROM continuumio/miniconda3

RUN mkdir -p project \
    && git clone signal-analysis project/

RUN cd signal-analysis \
    conda env update --file environment.yml --name ml_ds