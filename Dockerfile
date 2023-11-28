FROM continuumio/miniconda3

COPY . /projects/signal-analysis

WORKDIR /projects/signal-analysis

RUN conda env update --file environment.yml --name base

EXPOSE 8888