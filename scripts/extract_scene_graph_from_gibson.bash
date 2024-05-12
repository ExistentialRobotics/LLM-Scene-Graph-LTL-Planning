#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname $SCRIPT_DIR)"

set -e
set -x

erl-gibson-scene-graph --help

for env in Allensville Benevolence Collierville; do
    erl-gibson-scene-graph --mesh-file $PROJECT_DIR/data/mesh_$env.obj \
        --scene-graph-npz-file $PROJECT_DIR/data/3DSceneGraph_$env.npz \
        --output-dir $PROJECT_DIR/outputs/$env \
        --save-images \
        --save-video
done
