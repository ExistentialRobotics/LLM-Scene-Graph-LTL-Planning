# code inspired from https://github.com/RoboCoachTechnologies/GPT-Synthesizer/blob/master/gpt_synthesizer/parser.py
import ast
import re
import numpy as np
import yaml

GPT_SPOT_SYNTAX = {
    "NEGATION": "!",
    "IMPLY": "i",
    "AND": "&",
    "EQUAL": "e",
    "UNTIL": "U",
    "ALWAYS": "G",
    "EVENTUALLY": "F",
    "OR": "|",
}


GPT_SPOT_SYNTAX_INV = {value: key for (key, value) in GPT_SPOT_SYNTAX.items()}


def gpt_to_spot(gpt_str: str) -> tuple[str, dict]:
    """
    Convert LLM output to Spot syntax
    :param gpt_str: LLM output
    :return: Spot syntax, atomic proposition dictionary
    """
    tl_list = ast.literal_eval(gpt_str)
    ap_dict = dict()
    spot_desc = ""
    for tl in tl_list:
        if "enter" in tl or "reach" in tl:
            if tl not in ap_dict.keys():
                ap_dict[tl] = "p" + str(len(ap_dict) + 1)
            spot_desc += " {prop}".format(prop=ap_dict[tl])
        else:
            spot_desc += " {oper}".format(oper=GPT_SPOT_SYNTAX[tl])

    return spot_desc, ap_dict


def save_ap_desc(ap_dict: dict, file_name: str = "ap_desc.yaml"):
    """
    Save atomic proposition descriptions to a yaml file
    :param ap_dict: atomic proposition dictionary
    :param file_name: file name to save
    :return:
    """
    ap_desc = {}
    for k in ap_dict.keys():
        if k.startswith("enter"):
            ap_desc[ap_dict[k]] = {
                "type": "kEnterRoom",
                "uuid": int(k.split("_")[1].strip()[:-1]),
            }
        elif k.startswith("reach"):
            ap_desc[ap_dict[k]] = {
                "type": "kReachObject",
                "uuid": int(k.split("_")[1].strip()[:-1]),
                "reach_distance": -1.0,  # negative means not specified
            }
        else:
            raise ValueError("Unknown AP type: {}".format(k))
    print("Save AP descriptions to ap_desc.yaml")
    print(ap_desc)
    with open(file_name, "w") as f:
        yaml.dump(ap_desc, f)


def save_ap_desc_npz(ap_dict: dict, file_name: str = "ap_desc.npz"):
    ap_desc = {}
    for k in ap_dict.keys():
        ap_desc[ap_dict[k]] = k
    np.savez(file_name, **ap_desc)


def load_building_desc(yaml_file):
    with open(yaml_file, "r") as file:
        building_dict = yaml.load(file, Loader=yaml.FullLoader)

    building_desc = dict()
    for floor_key in building_dict["floors"].keys():
        floor = building_dict["floors"][floor_key]
        for room_key in floor["rooms"].keys():
            room = floor["rooms"][room_key]
            objects = dict()
            for obj_key in room["objects"].keys():
                obj = room["objects"][obj_key]
                objects[obj["name"] + "_" + str(obj["uuid"])] = "object_" + str(obj["uuid"])

            building_desc[room["name"] + "_" + str(room["uuid"])] = {
                "floor": str(floor_key),
                "id": "room_" + str(room["uuid"]),
                "objects": objects,
            }

    return building_desc


def list_building_desc(building_desc):
    list_str = ""
    for room in building_desc.keys():
        objects_str = "["

        for obj in building_desc[room]["objects"].keys():
            objects_str += "'" + obj + "': '" + building_desc[room]["objects"][obj] + "', "

        if len(objects_str) > 1:
            objects_str = objects_str[:-2] + "]"
        else:
            objects_str = "[]"

        list_str += """- {room_name}:
            - floor: {floor}
            - id: {room_id}
            - objects:
                {objects}\n""".format(
            room_name=room, floor=building_desc[room]["floor"], room_id=building_desc[room]["id"], objects=objects_str
        )

    return list_str


def find_env_elements(input_str):
    env_elements = set()
    room_ind_tuple_list = [(m.start(), m.end()) for m in re.finditer(r"room_\d+", input_str)]
    object_ind_tuple_list = [(m.start(), m.end()) for m in re.finditer(r"object_\d+", input_str)]

    if len(room_ind_tuple_list) > 0:
        for room_ind_tuple in room_ind_tuple_list:
            env_elements.add(input_str[room_ind_tuple[0] : room_ind_tuple[1]])

    if len(object_ind_tuple_list) > 0:
        for object_ind_tuple in object_ind_tuple_list:
            env_elements.add(input_str[object_ind_tuple[0] : object_ind_tuple[1]])

    env_elements = list(env_elements)
    env_elements_str = str(env_elements).replace("'", "")

    return env_elements, env_elements_str


def parse_syntax_error(syntax_error, ap_dict):
    ap_dict_inv = {value: key for (key, value) in ap_dict.items()}
    spot_formula_str = syntax_error.splitlines()[1][5:]
    gpt_formula = []
    for element in spot_formula_str.split(" "):
        if element in ap_dict_inv.keys():
            gpt_formula.append(ap_dict_inv[element])
        elif element in GPT_SPOT_SYNTAX_INV.keys():
            gpt_formula.append(GPT_SPOT_SYNTAX_INV[element])
        else:
            print("Error: syntax error is invalid!")

    indicator_str = syntax_error.splitlines()[2][5:]
    error_element_ind = spot_formula_str[: len(indicator_str)].count(" ")
    gpt_formula[error_element_ind] = "{error_element} --> INCORRECT".format(
        error_element=gpt_formula[error_element_ind]
    )

    error_str = syntax_error.splitlines()[3]

    return str(gpt_formula), error_str
