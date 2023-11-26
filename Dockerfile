FROM python:3.10-bookworm

WORKDIR /GEO

COPY ./requirement.txt /GEO/requirement.txt
RUN pip3 install -r /GEO/requirements.txt

# Install Tmux and openssh
RUN apt-get update && \
  apt install -y tmux && \
  apt install -y openssh-server