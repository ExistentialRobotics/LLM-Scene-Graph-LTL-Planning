import argparse
import os
import time
import xml.etree.ElementTree as ElementTree

import numpy as np
from openai import OpenAI
import spot
import yaml

from llm_planning.llm_heuristic.gen_automaton_state_desc import load_ap_desc
from llm_planning.llm_heuristic.gen_building_desc_from_yaml import load_building_yaml
from llm_planning.llm_heuristic.gen_task_desc_automaton_state import gen_task_state_desc
from llm_planning.llm_heuristic.gen_task_desc_automaton_state import build_graph_from_spot

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

output_example = "\n \
The main functions you can use are: \n \
move(room1, room2): move from room1 to room2 only if room1 is connected to room2. \n \
reach(room, object): reach an object only if the room has the object.  \n \
\n \
Each function call in your output is identified by the following tag: \n \
<command> Output a function call in a sequence </command> \n \
Your output contains those tags only. \n \
\n \
An example of your task is to visit the couch in the living room and then go to the kitchen. If you are in corridor 7, you can achieve the task by the following sequence of function calls: \n \
<command>move(7, 8)</command> \n \
<command>move(8, 10)</command> \n \
<command>reach(8, 38)</command> \n \
<command>move(10, 8)</command> \n \
<command>reach(8, 9)</command>"


def parse_function(function, locations):
    function = function.strip()
    id_str = None
    ret = None
    try:
        if function[:4] == "move":
            id_str = function[5:-1].split(",")
        elif function[:5] == "reach":
            id_str = function[6:-1].split(",")
        if id_str is not None:
            if len(id_str) == 2:
                ret = np.linalg.norm(
                    np.array(locations[int(id_str[0].strip())]) - np.array(locations[int(id_str[1].strip())])
                )
    except Exception as e:
        print("An exception has happened with parsing {}: \n {}".format(function, str(e)))
    return ret


def chatgpt_path(
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    room_uuid: int,
    automaton_state: int,
    accepting_state: int,
    building_desc: str,
    ap_desc: dict,
    graph: dict,
    edges: dict,
):
    start = time.time()
    gen_task = gen_task_state_desc(graph, edges, automaton_state, accepting_state, ap_desc)
    if len(gen_task) == 0:
        return [], 0
    user_msg = (
        "You are in the room with ID "
        + str(room_uuid)
        + ". Your task is to "
        + gen_task
        + ". How do you finish the task? Answer using move and reach function only."
        + " Use minimal number of function calls. No explanation."
    )
    print("User message: {}".format(user_msg))
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "system", "content": building_desc + output_example}, {"role": "user", "content": user_msg}],
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )
    end = time.time()
    tokens = response.usage.prompt_tokens + response.usage.completion_tokens
    print(
        "Time taken: {}s for {} prompt tokens and {} completion tokens".format(
            end - start, response.usage.prompt_tokens, response.usage.completion_tokens
        )
    )
    print("\n\n#############################################################################\n\n Output =\n")
    print(response.choices[0].message.content)

    result = "<response>\n" + response.choices[0].message.content + "\n</response>"
    root = ElementTree.fromstring(result)

    gpt_path = []
    for child in root.iter("*"):
        if child.tag == "response":
            continue
        gpt_path.append(child.text)
    print("Path from ChatGPT for room {} and dfa state {}: {} \n".format(room_uuid, automaton_state, gpt_path))
    print("\n\n#############################################################################\n\n")
    return gpt_path, tokens


def main():
    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.abspath(os.path.join(pkg_dir, "../../../../data"))
    building_desc_file = os.path.join(data_dir, "Collierville/building.yaml")
    ap_desc_file = os.path.join(data_dir, "Collierville/missions/1/ap_desc.npz")
    automaton_file = os.path.join(data_dir, "Collierville/missions/1/automaton.aut")
    task_desc_file = os.path.join(data_dir, "Collierville/missions/1/NL_instructions_uuid.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--building-desc-file",
        type=str,
        default=building_desc_file,
        help=f"Path to the building description file. Default: {building_desc_file}.",
    )
    parser.add_argument(
        "--ap-desc-file",
        type=str,
        default=ap_desc_file,
        help=f"Path to the atomic proposition description file in .npz format. Default: {ap_desc_file}",
    )
    parser.add_argument(
        "--automaton-file",
        type=str,
        default=automaton_file,
        help=f"Path to the automaton file in .aut format. Default: {automaton_file}",
    )
    parser.add_argument(
        "--task-desc-file",
        type=str,
        default=task_desc_file,
        help=f"Path to the file of natural language instructions with room and object UUIDs. Default: {task_desc_file}",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="llm_heuristic.yaml",
        help="Path to the output file in .yaml format. Default: llm_heuristic.yaml",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4",
        help=f"Model name for the LLM. Default: gpt-4.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help=f"Temperature parameter for the LLM. Default: 0.0.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=2048,
        help=f"Maximum number of tokens for the LLM. Default: 2048.",
    )
    args = parser.parse_args()
    building_desc_file = args.building_desc_file
    ap_desc_file = args.ap_desc_file
    automaton_file = args.automaton_file
    task_desc_file = args.task_desc_file
    output_file = args.output_file
    llm_model = args.llm_model
    llm_temperature = args.llm_temperature
    llm_max_tokens = args.llm_max_tokens

    # LOAD BUILDING
    building_desc, room_names, locations = load_building_yaml(building_desc_file)
    print("\nBuilding description:")
    print(building_desc)

    print("\nRoom names:")
    print(room_names)

    # LOAD TASK + AUTOMATON
    with open(task_desc_file) as f:
        task_desc = f.readlines()

    task = task_desc[0].replace("_", " with ID ")
    print("\nTask description:")
    print(task)

    spot_aut = spot.automaton(automaton_file)
    ap_desc = load_ap_desc(ap_desc_file)
    start_state, edges, graph, accepting_states = build_graph_from_spot(spot_aut, num_aps=len(ap_desc))
    print("start_state = {}".format(start_state))
    print("AP values for each edge: {}".format(edges))

    # GENERATE GPT PATH
    llm_heuristic_dict = {}
    with open(output_file, "w") as file:
        total_tokens = 0
        for room_uuid in room_names.keys():
            room_dict = {}
            for dfa_state in range(spot_aut.num_states()):
                if dfa_state in accepting_states:
                    continue
                for attempt in range(5):
                    success = True
                    try:
                        gpt_path, tokens = chatgpt_path(
                            llm_model,
                            llm_temperature,
                            llm_max_tokens,
                            room_uuid,
                            dfa_state,
                            accepting_states[0],
                            building_desc,
                            ap_desc,
                            graph,
                            edges,
                        )
                        print("\nattempt {} succeeded!".format(attempt))
                    except Exception as e:
                        print("\nAn exception has happened: \n {}".format(str(e)))
                        success = False
                        print("\nattempt {} failed! Sleep for 10 secs!".format(attempt))
                        time.sleep(10)
                    if success:
                        break
                room_dict[dfa_state] = gpt_path
                time.sleep(15)  # sleep to avoid exceeding the API rate limit
                total_tokens = total_tokens + tokens
            llm_heuristic_dict[room_uuid] = room_dict

        print(llm_heuristic_dict)
        yaml.dump(llm_heuristic_dict, file)
        print("Total tokens: {}".format(total_tokens))


if __name__ == "__main__":
    main()
