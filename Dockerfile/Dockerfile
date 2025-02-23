FROM nvcr.io/nvidia/pytorch:24.03-py3

ENV PYTHONUNBUFFERED=1
ARG DEBIAN_FRONTEND=noninteractive
# system
RUN apt-get update
RUN apt-get install -y build-essential apt-utils wget curl vim git pip \
    ca-certificates kmod \
    libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
    libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
    libeigen3-dev libxxf86vm-dev libembree-dev
RUN rm -rf /var/lib/apt/lists/*

