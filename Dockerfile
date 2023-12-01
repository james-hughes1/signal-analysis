FROM continuumio/miniconda3

RUN mkdir -p signal-analysis

COPY . /signal-analysis

WORKDIR /signal-analysis

RUN conda env update --file environment.yml --name base

EXPOSE 8888