FROM continuumio/miniconda3

RUN mkdir -p project \
    && git clone https://github.com/james-hughes1/signal-analysis signal-analysis/

RUN cd signal-analysis \
    conda env update --file environment.yml --name ml_ds