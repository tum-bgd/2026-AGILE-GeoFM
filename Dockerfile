FROM python:3.10-bookworm AS python-geofm

RUN pip install --upgrade pip

WORKDIR /GEO

# Install Tmux and openssh
RUN apt-get update && \
  apt-get install -y libgl1 && \
  apt install -y tmux

# Install requirements
COPY ./requirements.txt /GEO/requirements.txt
RUN pip install -r /GEO/requirements.txt

# Install GroundingDINO dependencies
COPY ./GroundingDINO /GEO/GroundingDINO
RUN pip install -e /GEO/GroundingDINO
