#!/usr/bin/env bash
#
# SOURCED FROM https://github.com/dusty-nv/jetson-inference/blob/master/docker/build.sh
#
# This script builds the jetson-inference docker container from source.
# It should be run from the root dir of the jetson-inference project:
#
#     $ cd /path/to/your/RealSenseInference
#     $ docker/build.sh
#
# Also you should set your docker default-runtime to nvidia:
#     https://github.com/dusty-nv/jetson-containers#docker-default-runtime
#

BASE_IMAGE=$1

# find L4T_VERSION
source tools/l4t-version.sh

if [ -z $BASE_IMAGE ]; then
	if [ $L4T_VERSION = "32.4.4" ]; then
		BASE_IMAGE="dustynv/jetson-inference:r32.4.4"
	elif [ $L4T_VERSION = "32.4.3" ]; then
		BASE_IMAGE="dustynv/jetson-inference:r32.4.3"
	else
		echo "cannot build jetson-inference docker container for L4T R$L4T_VERSION"
		echo "please upgrade to the latest JetPack, or build jetson-inference natively"
		exit 1
	fi
fi

echo "BASE_IMAGE=$BASE_IMAGE"
echo "TAG=rs-inference:r$L4T_VERSION"
# build the container
sudo docker build -t rs-inference:r$L4T_VERSION -f docker/Dockerfile.rsi \
    --build-arg BASE_IMAGE=$BASE_IMAGE .
