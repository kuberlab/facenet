FROM ubuntu:16.04

# Install necessary packages for the installer
RUN apt-get update && apt-get install -y \
    build-essential \
    sed \
    sudo \
    tar \
    udev \
    wget \
    git \
    python-dev \
    python3.5-dev

# Build NCSDK
RUN git clone https://github.com/movidius/ncsdk.git && cd ncsdk/ && \
  git checkout ncsdk2 && \
  make install
  #cd ../ && rm -rf ncsdk

RUN apt-get install python-opencv ffmpeg -y

RUN pip install tensorflow==1.7.0 scipy sklearn opencv-python requests imageio Pillow && pip install -U numpy && \
  pip3 install tensorflow==1.7.0 scipy sklearn opencv-python requests imageio && pip3 install -U numpy

ENV PYTHONPATH /opt/movidius/caffe/python

RUN pip install --no-cache-dir 'git+https://github.com/kuberlab/python-mlboardclient.git'
# Install caffe
#RUN apt-get install -y --no-install-recommends \
#      build-essential \
#      cmake \
#      git \
#      libgoogle-glog-dev \
#      libgtest-dev \
#      libgflags-dev \
#      libiomp-dev \
#      libleveldb-dev \
#      liblmdb-dev \
#      libopencv-dev \
#      libopenmpi-dev \
#      libsnappy-dev \
#      libprotobuf-dev \
#      openmpi-bin \
#      openmpi-doc \
#      protobuf-compiler \
#      python-dev \
#      python-pip && \
#  pip install \
#      future \
#      numpy \
#      protobuf
#
#RUN git clone --recursive https://github.com/pytorch/pytorch.git && cd pytorch && \
#  git submodule update --init && \
#  mkdir build && cd build && \
#  cmake .. && \
#  make -j"$(nproc)" install && \
#  ldconfig && \
#  make clean && \
#  rm -rf * && \
## Build for python3 also
#  cmake \
#   -DPYTHON_INCLUDE_DIR=$(python3 -c 'from distutils import sysconfig; print(sysconfig.get_python_inc())') \
#   -DPYTHON_EXECUTABLE=/usr/bin/python3 \
#   .. && \
#  make -j"$(nproc)" install && \
#  ldconfig && \
#  make clean && \
#  cd .. && \
#  rm -rf build
#  cd ../.. && rm -rf pytorch
