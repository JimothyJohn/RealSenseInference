#!/usr/bin/env bash

docker/run.sh \
    -v $(pwd)/rsinfer:/rsinfer \
    -v $(pwd)/data/networks:/networks \
    -v $(pwd)/data/images:/images \
    --run python /rsinfer \
    --model /networks/dogs.eim \
    --outdir /images
