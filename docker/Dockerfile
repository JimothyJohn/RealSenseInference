# Start from base image
ARG BASE_IMAGE=dustynv/jetson-inference:r32.7.1
FROM ${BASE_IMAGE}

## RealSense Install
# Update and install prereqs
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    git \
    libusb-1.0-0-dev \
    libssl-dev \
    libgtk-3-dev \
    pkg-config \
    python3-dev \
    python3-lxml \
    python3-pip \
    python3-six \
    python3-tk \
    xorg-dev \
    libatlas-base-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev


# Install librealsense from source
RUN cd / && git clone https://github.com/IntelRealSense/librealsense.git \
    && cd /librealsense \
    && sh ./scripts/setup_udev_rules.sh \
    && mkdir build \
    && cd /librealsense/build \
    && cmake ../ \
        -DCMAKE_BUILD_TYPE=release \
        -DFORCE_RSUSB_BACKEND=false \
        -DBUILD_WITH_CUDA=true \
        -DBUILD_PYTHON_BINDINGS:bool=true \
        -DPYTHON_EXECUTABLE=/usr/bin/python \
    && make -j$(($(nproc)-1)) && make install

COPY requirements.txt /
RUN pip install -U pip && \
    pip install -r /requirements.txt

# Add to path, return, and clean up
# ENV PYTHONPATH="/usr/local/lib/python3.6:$PYTHONPATH"
