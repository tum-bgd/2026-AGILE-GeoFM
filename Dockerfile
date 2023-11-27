FROM python:3.10-bookworm AS python-geofm

RUN pip install --upgrade pip

WORKDIR /GEO

# Install Tmux and openssh
RUN apt-get update && \
  apt install -y tmux && \
  apt install -y openssh-server

COPY ./requirements.txt /GEO/requirements.txt
RUN pip3 install -r /GEO/requirements.txt