#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname $SCRIPT_DIR)"

set -e
set -x

erl-nl2ltl --help

for env in Allensville Benevolence Collierville; do
    for mission_idx in 1 2 3 4 5; do
        OUTPUT_DIR=$PROJECT_DIR/outputs/$env/missions/$mission_idx
        mkdir -p $OUTPUT_DIR
        cat $PROJECT_DIR/data/$env/missions/$mission_idx/NL_instructions.txt | xargs -I {} \
        erl-nl2ltl --building-file $PROJECT_DIR/data/$env/building.yaml \
            --instruction {} \
            --output-dir $OUTPUT_DIR
    done
done
