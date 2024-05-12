#!/usr/bin/env bash

set -e
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname $SCRIPT_DIR)"
EXECUTABLE="$PROJECT_DIR/build/cpp/src/cpp/amra_ltl_scene_graph"
if [[ ! -f $EXECUTABLE ]]; then
    echo "$EXECUTABLE does not exist. Please build the C++ source at first."
fi

$EXECUTABLE --help

DATA_DIR=$PROJECT_DIR/data

OUTPUT_DIR=$PROJECT_DIR/outputs/amra_ltl_scene_graph
mkdir -p $OUTPUT_DIR

for ENV in Allensville Benevolence Collierville; do
    INITIAL_POSITION_YAML=$DATA_DIR/$ENV/initial_positions.yaml
    source $DATA_DIR/$ENV/radius_config.bash
    for MISSION_IDX in 1 2 3 4 5; do
        for POSITION_IDX in 0 1 2 3 4; do
            INIT_GRID_X=$(python -c "import yaml; print(yaml.safe_load(open('$INITIAL_POSITION_YAML', 'r'))[$POSITION_IDX][0])")
            INIT_GRID_Y=$(python -c "import yaml; print(yaml.safe_load(open('$INITIAL_POSITION_YAML', 'r'))[$POSITION_IDX][1])")
            INIT_GRID_Z=$(python -c "import yaml; print(yaml.safe_load(open('$INITIAL_POSITION_YAML', 'r'))[$POSITION_IDX][2])")
            echo "$ENV: mission $MISSION_IDX, initial_position: $INIT_GRID_X, $INIT_GRID_Y, $INIT_GRID_Z"
            $EXECUTABLE \
                --output-dir $OUTPUT_DIR/$ENV/$MISSION_IDX/ALL/$POSITION_IDX \
                --scene-graph-file $DATA_DIR/$ENV/building.yaml \
                --map-data-dir $DATA_DIR/$ENV \
                --automaton-file $DATA_DIR/$ENV/missions/$MISSION_IDX/automaton.aut \
                --ap-file $DATA_DIR/$ENV/missions/$MISSION_IDX/ap_desc.yaml \
                --llm-heuristic-file $DATA_DIR/$ENV/missions/$MISSION_IDX/llm_heuristic.yaml \
                --init-grid-x $INIT_GRID_X \
                --init-grid-y $INIT_GRID_Y \
                --init-grid-z $INIT_GRID_Z \
                --max-level kFloor \
                --ltl-heuristic-config 11111 \
                --llm-heuristic-config 01111 \
                --robot-radius $ROBOT_RADIUS \
                --object-reach-radius $OBJECT_REACH_RADIUS
            $EXECUTABLE \
                --output-dir $OUTPUT_DIR/$ENV/$MISSION_IDX/NO_LLM/$POSITION_IDX \
                --scene-graph-file $DATA_DIR/$ENV/building.yaml \
                --map-data-dir $DATA_DIR/$ENV \
                --automaton-file $DATA_DIR/$ENV/missions/$MISSION_IDX/automaton.aut \
                --ap-file $DATA_DIR/$ENV/missions/$MISSION_IDX/ap_desc.yaml \
                --llm-heuristic-file $DATA_DIR/$ENV/missions/$MISSION_IDX/llm_heuristic.yaml \
                --init-grid-x $INIT_GRID_X \
                --init-grid-y $INIT_GRID_Y \
                --init-grid-z $INIT_GRID_Z \
                --max-level kFloor \
                --ltl-heuristic-config 11111 \
                --llm-heuristic-config 00000 \
                --robot-radius $ROBOT_RADIUS \
                --object-reach-radius $OBJECT_REACH_RADIUS
            $EXECUTABLE \
                --output-dir $OUTPUT_DIR/$ENV/$MISSION_IDX/A_STAR/$POSITION_IDX \
                --scene-graph-file $DATA_DIR/$ENV/building.yaml \
                --map-data-dir $DATA_DIR/$ENV \
                --automaton-file $DATA_DIR/$ENV/missions/$MISSION_IDX/automaton.aut \
                --ap-file $DATA_DIR/$ENV/missions/$MISSION_IDX/ap_desc.yaml \
                --llm-heuristic-file $DATA_DIR/$ENV/missions/$MISSION_IDX/llm_heuristic.yaml \
                --init-grid-x $INIT_GRID_X \
                --init-grid-y $INIT_GRID_Y \
                --init-grid-z $INIT_GRID_Z \
                --max-level kOcc \
                --ltl-heuristic-config 11 \
                --llm-heuristic-config 00 \
                --robot-radius $ROBOT_RADIUS \
                --object-reach-radius $OBJECT_REACH_RADIUS
        done
    done
done

# ablation study
ENV=Benevolence
INITIAL_POSITION_YAML=$DATA_DIR/$ENV/initial_positions.yaml
source $DATA_DIR/$ENV/radius_config.bash
for MISSION_IDX in 1 2 3 4 5; do
    for POSITION_IDX in 0 1 2 3 4; do
        INIT_GRID_X=$(python -c "import yaml; print(yaml.safe_load(open('$INITIAL_POSITION_YAML', 'r'))[$POSITION_IDX][0])")
        INIT_GRID_Y=$(python -c "import yaml; print(yaml.safe_load(open('$INITIAL_POSITION_YAML', 'r'))[$POSITION_IDX][1])")
        INIT_GRID_Z=$(python -c "import yaml; print(yaml.safe_load(open('$INITIAL_POSITION_YAML', 'r'))[$POSITION_IDX][2])")
        echo "$ENV: mission $MISSION_IDX, initial_position: $INIT_GRID_X, $INIT_GRID_Y, $INIT_GRID_Z"
        $EXECUTABLE \
            --output-dir $OUTPUT_DIR/$ENV/$MISSION_IDX/FLR/$POSITION_IDX \
            --scene-graph-file $DATA_DIR/$ENV/building.yaml \
            --map-data-dir $DATA_DIR/$ENV \
            --automaton-file $DATA_DIR/$ENV/missions/$MISSION_IDX/automaton.aut \
            --ap-file $DATA_DIR/$ENV/missions/$MISSION_IDX/ap_desc.yaml \
            --llm-heuristic-file $DATA_DIR/$ENV/missions/$MISSION_IDX/llm_heuristic.yaml \
            --init-grid-x $INIT_GRID_X \
            --init-grid-y $INIT_GRID_Y \
            --init-grid-z $INIT_GRID_Z \
            --max-level kFloor \
            --ltl-heuristic-config 11111 \
            --llm-heuristic-config 00001 \
            --robot-radius $ROBOT_RADIUS \
            --object-reach-radius $OBJECT_REACH_RADIUS
        $EXECUTABLE \
            --output-dir $OUTPUT_DIR/$ENV/$MISSION_IDX/ROOM/$POSITION_IDX \
            --scene-graph-file $DATA_DIR/$ENV/building.yaml \
            --map-data-dir $DATA_DIR/$ENV \
            --automaton-file $DATA_DIR/$ENV/missions/$MISSION_IDX/automaton.aut \
            --ap-file $DATA_DIR/$ENV/missions/$MISSION_IDX/ap_desc.yaml \
            --llm-heuristic-file $DATA_DIR/$ENV/missions/$MISSION_IDX/llm_heuristic.yaml \
            --init-grid-x $INIT_GRID_X \
            --init-grid-y $INIT_GRID_Y \
            --init-grid-z $INIT_GRID_Z \
            --max-level kFloor \
            --ltl-heuristic-config 11111 \
            --llm-heuristic-config 00010 \
            --robot-radius $ROBOT_RADIUS \
            --object-reach-radius $OBJECT_REACH_RADIUS
        $EXECUTABLE \
            --output-dir $OUTPUT_DIR/$ENV/$MISSION_IDX/OBJ/$POSITION_IDX \
            --scene-graph-file $DATA_DIR/$ENV/building.yaml \
            --map-data-dir $DATA_DIR/$ENV \
            --automaton-file $DATA_DIR/$ENV/missions/$MISSION_IDX/automaton.aut \
            --ap-file $DATA_DIR/$ENV/missions/$MISSION_IDX/ap_desc.yaml \
            --llm-heuristic-file $DATA_DIR/$ENV/missions/$MISSION_IDX/llm_heuristic.yaml \
            --init-grid-x $INIT_GRID_X \
            --init-grid-y $INIT_GRID_Y \
            --init-grid-z $INIT_GRID_Z \
            --max-level kFloor \
            --ltl-heuristic-config 11111 \
            --llm-heuristic-config 00100 \
            --robot-radius $ROBOT_RADIUS \
            --object-reach-radius $OBJECT_REACH_RADIUS
        $EXECUTABLE \
            --output-dir $OUTPUT_DIR/$ENV/$MISSION_IDX/OCC/$POSITION_IDX \
            --scene-graph-file $DATA_DIR/$ENV/building.yaml \
            --map-data-dir $DATA_DIR/$ENV \
            --automaton-file $DATA_DIR/$ENV/missions/$MISSION_IDX/automaton.aut \
            --ap-file $DATA_DIR/$ENV/missions/$MISSION_IDX/ap_desc.yaml \
            --llm-heuristic-file $DATA_DIR/$ENV/missions/$MISSION_IDX/llm_heuristic.yaml \
            --init-grid-x $INIT_GRID_X \
            --init-grid-y $INIT_GRID_Y \
            --init-grid-z $INIT_GRID_Z \
            --max-level kFloor \
            --ltl-heuristic-config 11111 \
            --llm-heuristic-config 01000 \
            --robot-radius $ROBOT_RADIUS \
            --object-reach-radius $OBJECT_REACH_RADIUS
    done
done
