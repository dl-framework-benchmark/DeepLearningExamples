#!/bin/bash
CURRENT_USER=""
# CURRENT_USER="-u $(id -u):$(id -g)"
set +x
# launch_mode=${1:-"local"}
CMD="bert:latest bash"
CMD="--privileged --network=host $CMD"
PORT=""
# PORT=" -p 8118:8118"
EXTRA_VOL=""
EXTRA_VOL+=" -v $PWD:/workspace/bert"
EXTRA_VOL+=" -v $PWD/results:/results"
# EXTRA_VOL+=" -v $PWD/checkpoints:/checkpoints"
EXTRA_VOL+=" -v /dataset:/dataset"
EXTRA_VOL+=" -v $HOME:/home/caishenghang"
docker run -it --rm \
  --runtime=nvidia \
  $PORT \
  $CURRENT_USER \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  $EXTRA_VOL \
  $CMD
