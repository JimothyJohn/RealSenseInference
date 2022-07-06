#!/usr/bin/env bash
#
# This script builds the jetson-inference docker container from source.
# It should be run from the root dir of the jetson-inference project:
#
#     $ cd /path/to/your/jetson-inference
#     $ docker/build.sh
#
# Also you should set your docker default-runtime to nvidia:
#     https://github.com/dusty-nv/jetson-containers#docker-default-runtime
#

TAG=rsinference:latest

# sanitize workspace (so extra files aren't added to the container)
rm -rf python/training/classification/data/*
rm -rf python/training/classification/models/*

rm -rf python/training/detection/ssd/data/*
rm -rf python/training/detection/ssd/models/*
	
# build the container
sudo docker build -t $TAG -f docker/Dockerfile .
