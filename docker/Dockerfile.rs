# Start from base image
FROM nvcr.io/nvidia/l4t-base:r32.4.3

# Housekeeping
ARG DEBIAN_FRONTEND=noninteractive

# Update and install prereqs
RUN apt-get update
RUN apt-get -y install cmake xorg-dev libusb-1.0-0-dev git python3-dev python3-pip python3-tk python3-lxml python3-six

# Install librealsense from source
WORKDIR /
RUN git clone https://github.com/IntelRealSense/librealsense.git
WORKDIR /librealsense
RUN mkdir build
WORKDIR /librealsense/build
RUN cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3
RUN make -j4 && make install

# Add to path, return, and clean up
RUN echo 'export PYTHONPATH=/usr/local/lib/python3.6:$PYTHONPATH' >> ~/.bashrc

WORKDIR /
