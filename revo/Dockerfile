FROM python:3.9-slim-buster as production

RUN apt-get update && apt-get install -y \
    gcc python3-dev \
    git

# Create user with non-root privileges
ARG UID=1001
ARG GID=1002

ENV PYTHONPATH="${PYTHONPATH}:/revo/"

RUN mkdir revo
WORKDIR /revo

COPY ./revo /revo

RUN pip install --upgrade pip
RUN pip install -r /revo/requirements.txt
 
COPY ./entry_scripts/production.sh /revo/entry_scripts/production.sh
RUN chmod +x -R ./entry_scripts

# Show service name in bash prompt
RUN echo 'PS1="\e[0;31m[revo]\e[m $PS1"' >> /home/adsp/.bashrc

ENTRYPOINT ["/bin/sh", "-c", "/revo/entry_scripts/production.sh"]

# Multi-stage build
FROM production as development
USER root

RUN mkdir /revo/notebooks /revo/static_test
COPY ./revo/notebooks /revo/notebooks
COPY ./revo/requirements_dev.txt /revo/requirements_dev.txt
COPY ./static_test/bash_aliases.txt /home/user/.bash_aliases
COPY ./static_test /revo/static_test

RUN pip install -r /src/requirements/requirements_dev.txt
COPY ./entry_scripts/development.sh /src/entry_scripts/development.sh
RUN chmod +x -R ./entry_scripts

EXPOSE 8888
ENTRYPOINT ["/bin/sh", "-c", "/src/entry_scripts/development.sh"]
