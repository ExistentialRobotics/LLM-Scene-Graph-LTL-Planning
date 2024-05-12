#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname $SCRIPT_DIR)"

set -e
set -x

erl-llm-heuristic --help

for ENV in Allensville Benevolence Collierville; do
    for MISSION_IDX in 1 2 3 4 5; do
        MISSION_DIR=$PROJECT_DIR/data/$ENV/missions/$MISSION_IDX
        OUTPUT_DIR=$PROJECT_DIR/outputs/$ENV/missions/$MISSION_IDX
        erl-llm-heuristic --building-file $PROJECT_DIR/data/$ENV/building.yaml \
            --ap-desc-file $MISSION_DIR/ap_desc.npz \
            --automaton-file $MISSION_DIR/automaton.aut \
            --task-desc-file $MISSION_DIR/NL_instructions_uuid.txt \
            --output-file $OUTPUT_DIR/llm_heuristic.yaml
    done
done
