FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV BUILD_WITH_CUDA=True
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV CUDA_HOME /usr/local/cuda-11.7/

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Tmux
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev git && \
    apt install -y tmux

WORKDIR /GEOtmp
RUN pip install pip==22.3.1

# Install requirements
COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt

# Install GroundingDINO dependencies
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
RUN pip install -e ./GroundingDINO

WORKDIR /GEO
