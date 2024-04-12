FROM python:3.9-slim-buster as production

USER root

ARG UID=1001
ARG GID=1002

ENV UID=1001
ENV GID=1002

# Install some basic packages
RUN apt-get update && apt-get install -y \
    gcc python3-dev \
    git

WORKDIR /revo

ENV PYTHONPATH="${PYTHONPATH}:/revo/"

COPY ./revo/data /revo/data
COPY ./revo/data_modified /revo/data_modified
COPY ./revo/dimension /revo/dimension
COPY ./revo/modules /revo/modules
COPY ./revo/requirements.txt /revo/requirements.txt 

RUN pip install --upgrade pip
RUN pip install -r /revo/requirements.txt
 
COPY ./entry_scripts/production.sh /revo/entry_scripts/production.sh
RUN chmod +x -R ./entry_scripts

# Show service name in bash prompt
RUN echo 'PS1="\e[0;31m[revo]\e[m $PS1"' >> /root/.bashrc

ENTRYPOINT ["/bin/sh", "-c", "/revo/entry_scripts/production.sh"]

# Multi-stage build
FROM production as development

RUN mkdir /revo/notebooks /revo/static_test
COPY ./revo/notebooks /revo/notebooks
COPY ./revo/requirements_dev.txt /revo/requirements_dev.txt
COPY ./static_test/bash_aliases.txt /root/.bash_aliases
COPY ./static_test /revo/static_test

RUN pip install -r /revo/requirements_dev.txt
COPY ./entry_scripts/development.sh /revo/entry_scripts/development.sh
RUN chmod +x -R ./entry_scripts

EXPOSE 8888
ENTRYPOINT ["/bin/sh", "-c", "/revo/entry_scripts/development.sh"]
