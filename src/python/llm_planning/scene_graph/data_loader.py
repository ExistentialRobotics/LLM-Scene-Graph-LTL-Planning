import cv2
import os
import pickle
import yaml

import numpy as np

import llm_planning.scene_graph.mesh_tools as mt
from llm_planning.scene_graph.mesh_tools import CHAR_DICT


class Building:
    def __init__(self):
        # Building 3D Scene Graph attributes
        self.floor_area = None  # 2D floor area in sq.meters
        self.function = None  # function of building
        self.name = None  # name of gibson model
        self.num_floors = None  # number of floors in the building
        self.num_rooms = None  # number of rooms in the building
        self.reference_point = None  # building reference point
        self.size = np.empty(3)  # 3D Size of building
        self.volume = None  # 3D volume of building computed from 3D convex hull (cubic meters)

        # downstream graph layer
        self.rooms = {}

        # floor grid map
        self.floor_maps = {}

    def set_attribute(self, value, attribute):
        # Set a building attribute
        if attribute not in self.__dict__.keys():
            print("Unknown building attribute: {}".format(attribute))
            return
        self.__dict__[attribute] = value

    def get_attribute(self, attribute):
        # Get a building attribute
        if attribute not in self.__dict__.keys():
            print("Unknown building attribute: {}".format(attribute))
            return -1
        return self.__dict__[attribute]

    def convert_to_dict(self):
        ret_dict = {}
        for key in self.__dict__.keys():
            if key in ["rooms", "floor_maps"]:
                continue

            elif key in ["size", "reference_point"]:
                ret_list = []
                for elem in self.__dict__[key]:
                    ret_list.append(elem.item())

                ret_dict[key] = ret_list

            else:
                ret_dict[key] = self.__dict__[key]

        ret_dict["rooms"] = {}
        for room_key in self.rooms.keys():
            ret_dict["rooms"][str(int(room_key))] = self.rooms[room_key].convert_to_dict()

        return ret_dict


class Room:
    def __init__(self):
        # Room 3D Scene Graph attributes
        self.floor_area = None  # 2D floor area in sq.meters
        self.floor_number = None  # index of floor that contains the space
        self.id = None  # unique space id per building
        self.location = np.empty((3))  # 3D coordinates of room center's location
        self.scene_category = None  # function of this room
        self.size = np.empty((3))  # 3D Size of room
        self.volume = None  # 3D volume of room computed from 3D convex hull (cubic meters)
        self.parent_building = None  # parent building that contains this room

        # downstream graph layer
        self.objects = {}

        # room connections
        self.connections = set()  # needs to be sorted out

    def set_attribute(self, value, attribute):
        # Set a room attribute
        if attribute not in self.__dict__.keys():
            print("Unknown room attribute: {}".format(attribute))
            return
        self.__dict__[attribute] = value

    def get_attribute(self, attribute):
        # Get a room attribute
        if attribute not in self.__dict__.keys():
            print("Unknown room attribute: {}".format(attribute))
            return -1
        return self.__dict__[attribute]

    def convert_to_dict(self):
        ret_dict = {}
        for key in self.__dict__.keys():
            if key == "objects":
                continue

            elif key in ["size", "location", "connections"]:
                ret_list = []
                for elem in self.__dict__[key]:
                    ret_list.append(elem.item())

                ret_dict[key] = ret_list

            else:
                ret_dict[key] = self.__dict__[key]

        ret_dict["objects"] = {}
        for object_key in self.objects.keys():
            ret_dict["objects"][str(int(object_key))] = self.objects[object_key].convert_to_dict()

        return ret_dict


class Object:
    def __init__(self):
        # Object 3D Scene Graph attributes
        self.action_affordance = None  # list of possible actions
        self.floor_area = None  # 2D floor area in sq.meters
        self.surface_coverage = None  # total surface coverage in sq.meters
        self.class_ = None  # object label
        self.id = None  # unique object id per building
        self.location = np.empty((3))  # 3D coordinates of object center's location
        self.material = None  # list of main object materials
        self.size = np.empty((3))  # 3D Size of object
        self.tactile_texture = None  # main tactile texture (can be None)
        self.visual_texture = None  # main visible texture (can be None)
        self.volume = None  # 3D volume of object computed from 3D convex hull (cubic meters)
        self.parent_room = None  # parent room that contains this object

    def set_attribute(self, value, attribute):
        # Set an object attribute
        if attribute not in self.__dict__.keys():
            print("Unknown object attribute: {}".format(attribute))
            return
        self.__dict__[attribute] = value

    def get_attribute(self, attribute):
        # Get an object attribute
        if attribute not in self.__dict__.keys():
            print("Unknown object attribute: {}".format(attribute))
            return -1
        return self.__dict__[attribute]

    def convert_to_dict(self):
        ret_dict = {}
        for key in self.__dict__.keys():
            if key in [
                "size",
                "location",
            ]:
                ret_list = []
                for elem in self.__dict__[key]:
                    ret_list.append(elem.item())

                ret_dict[key] = ret_list

            elif key == "surface_coverage":
                ret_dict[key] = self.__dict__[key].item()

            else:
                ret_dict[key] = self.__dict__[key]

        return ret_dict


def load_scene_graph_file(npz_path, building):
    # Load 3D Scene Graph data in the npz file
    data = np.load(npz_path, allow_pickle=True)["output"].item()

    # set building attributes
    for key in data["building"].keys():
        if key in [
            "object_inst_segmentation",
            "room_inst_segmentation",
            "object_voxel_occupancy",
            "room_voxel_occupancy",
            "gibson_split",
            "id",
            "num_cameras",
            "num_objects",
            "voxel_size",
            "voxel_centers",
            "voxel_resolution",
        ]:
            continue
        building.set_attribute(data["building"][key], key)

    # set room attributes
    unique_rooms = np.unique(data["building"]["room_inst_segmentation"])
    for room_id in unique_rooms:
        if room_id == 0:
            continue
        building.rooms[room_id] = Room()
        for key in data["room"][room_id].keys():
            if key in ["inst_segmentation", "voxel_occupancy"]:
                continue
            building.rooms[room_id].set_attribute(data["room"][room_id][key], key)

    # set object attributes
    unique_objects = np.unique(data["building"]["object_inst_segmentation"])
    for object_id in unique_objects:
        if object_id == 0:
            continue

        parent_room = data["object"][object_id]["parent_room"]
        building.rooms[parent_room].objects[object_id] = Object()
        for key in data["object"][object_id].keys():
            if key in ["inst_segmentation", "voxel_occupancy"]:
                continue
            building.rooms[parent_room].objects[object_id].set_attribute(data["object"][object_id][key], key)

    return building


def load_mesh_file(mesh_path, npz_path, building):
    # Load 3D mesh data in the obj file
    mesh = mt.load_mesh(mesh_path)
    vertices = mesh.vertices
    faces = mesh.faces

    for floor_num in range(building.num_floors):
        floor_char = CHAR_DICT[floor_num]
        face_pos, face_room_cat, face_obj_cat = mt.extract_face_segmentation(
            npz_path, vertices, faces, building, floor_char
        )
        floor_grid_map = mt.get_grid_map(face_pos, face_room_cat, face_obj_cat, 0.1)
        floor_grid_map = mt.cleanup_map(floor_grid_map, kernel_size=3)
        building.floor_maps[floor_char] = floor_grid_map
        building = mt.find_room_connections(building, floor_char, kernel_size=5, min_intersection=5)

    return building


def load_3DSceneGraph(environment, data_path):
    # Load 3D SceneGraph attributes
    # model: name of Gibson model
    # data_path : location of folder with annotations and mesh
    building = Building()
    npz_path = os.path.join(data_path, "3DSceneGraph_" + environment + ".npz")
    mesh_path = os.path.join(data_path, "mesh_" + environment + ".obj")
    building = load_scene_graph_file(npz_path, building)
    building = load_mesh_file(mesh_path, npz_path, building)
    return building


def get_edges(input_struct, level="building"):
    edges = []
    if level == "building":
        for room_key in input_struct.rooms.keys():
            edges.append(
                (
                    input_struct.name,
                    input_struct.rooms[room_key].scene_category.replace(" ", "_")
                    + "_"
                    + str(input_struct.rooms[room_key].id),
                )
            )
            edges.extend(get_edges(input_struct.rooms[room_key], level="room"))

    if level == "room":
        for object_key in input_struct.objects.keys():
            edges.append(
                (
                    input_struct.scene_category.replace(" ", "_") + "_" + str(input_struct.id),
                    input_struct.objects[object_key].class_.replace(" ", "_")
                    + "_"
                    + str(input_struct.objects[object_key].id),
                )
            )

    return edges


def print_json(building):
    print(str(building.convert_to_dict()).replace("'", '"').replace("None", '""'))


def dump_edges(building):
    edges = get_edges(building)
    # Dump edge list in Graphviz DOT format
    print("strict digraph tree {")
    for row in edges:
        print("    {0} -> {1};".format(*row))
    print("}")


def save_as_yaml(building, filename):
    building_dict = building.convert_to_dict()

    if not os.path.exists("../../data/{filename}".format(filename=filename)):
        os.mkdir("../../data/{filename}".format(filename=filename))

    with open(r"../../data/{filename}/{filename}.yaml".format(filename=filename), "w") as file:
        yaml.dump(building_dict, file)

    if not os.path.exists("../../data/{filename}/floor_maps".format(filename=filename)):
        os.mkdir("../../data/{filename}/floor_maps".format(filename=filename))

    for floor_name in building.floor_maps.keys():
        floor_map = building.floor_maps[floor_name]
        cv2.imwrite(
            "../../data/{filename}/floor_maps/{floor_name}_room.png".format(filename=filename, floor_name=floor_name),
            floor_map.get_room_map().astype(np.uint8),
        )
        cv2.imwrite(
            "../../data/{filename}/floor_maps/{floor_name}_object.png".format(filename=filename, floor_name=floor_name),
            floor_map.get_cat_map().astype(np.uint8),
        )


if __name__ == "__main__":
    data_path = os.getcwd() + "/../../data/"
    pickle_path = os.path.join(data_path, "v1_pkl")
    os.makedirs(pickle_path, exist_ok=True)
    environment = "Benevolence"  # Set the environment name here

    filename = os.path.join(pickle_path, "{env}.pkl".format(env=environment))
    if os.path.isfile(filename):
        with open(filename, "rb") as file:
            building = pickle.load(file)
    else:
        building = load_3DSceneGraph(environment, data_path)
        with open(filename, "wb") as file:
            pickle.dump(building, file)

    mt.draw_room_connections(building)
    mt.show_cat_map(building)
    save_as_yaml(building, environment)
