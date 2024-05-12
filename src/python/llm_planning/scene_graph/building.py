import os
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

from llm_planning.scene_graph.floor import Base, Floor
from llm_planning.scene_graph.special_object_categories import SOC


class Building(Base):
    def __init__(self):
        super().__init__()
        # Building 3D Scene Graph attributes
        self.id: int = -1  # unique building id
        self.parent_id: int = -1  # parent id
        self.floor_area: float = -1  # 2D floor area in sq.meters
        self.function: str = ""  # function of building
        self.name: str = ""  # name of gibson model
        self.num_floors: int = -1  # number of floors in the building
        self.num_rooms: int = -1  # number of rooms in the building
        self.reference_point: np.ndarray = np.zeros(3)  # building reference point
        self.size: np.ndarray = np.zeros(3)  # 3D Size of building
        self.volume: float = 0  # 3D volume of building computed from 3D convex hull (cubic meters)
        self.floors: Dict[int, Floor] = {}  # floor number to floor object
        self.room_id_to_floor_num: Dict[int, int] = {}  # room id to floor number
        self.object_id_to_room_id: Dict[int, int] = {}  # object id to room id

    def set_attribute(self, attribute, value):
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
        ret_dict = super().convert_to_dict()
        ret_dict["type"] = "kBuilding"
        for key in self.__dict__.keys():
            if key in ["size", "reference_point"]:
                ret_dict[key] = self.__dict__[key].tolist()
            elif key == "floors":
                floor_dict = {}
                for floor_num, floor in self.floors.items():
                    floor_dict[floor_num] = floor.convert_to_dict()
                ret_dict[key] = floor_dict
            elif key == "room_id_to_floor_num" or key == "object_id_to_room_id":
                continue
            else:
                ret_dict[key] = self.__dict__[key]

        return ret_dict

    def load_from_dict(self, building_dict: dict):
        assert building_dict["type"] == "kBuilding", "Not a building dict!"
        for key, value in building_dict.items():
            if key == "floors":
                for floor_num, floor_dict in value.items():
                    floor = Floor()
                    floor.load_from_dict(floor_dict)
                    self.floors[floor_num] = floor
            elif key in ["room_id_to_floor_num", "object_id_to_room_id"]:
                continue
            elif key in ["size", "reference_point"]:
                self.__dict__[key] = np.array(value)
            else:
                self.__dict__[key] = value
        self.room_id_to_floor_num = {}
        self.object_id_to_room_id = {}
        for floor_num, floor in self.floors.items():
            for room_id, room in floor.rooms.items():
                self.room_id_to_floor_num[room_id] = floor_num
                for object_id, obj in room.objects.items():
                    self.object_id_to_room_id[object_id] = room_id

    def load(self, load_path: str):
        yaml_file = os.path.join(load_path, f"building.yaml")
        with open(yaml_file, "r") as file:
            building_dict = yaml.load(file, Loader=yaml.FullLoader)
        self.load_from_dict(building_dict)
        for floor_num, floor_dict in building_dict["floors"].items():
            room_map_path = os.path.join(load_path, floor_dict["room_map"])
            cat_map_path = os.path.join(load_path, floor_dict["cat_map"])
            room_map = cv2.imread(room_map_path, cv2.IMREAD_GRAYSCALE).astype(np.int32) + SOC.NA
            cat_map = cv2.imread(cat_map_path, cv2.IMREAD_GRAYSCALE).astype(np.int32) + SOC.NA
            floor = self.floors[floor_num]
            floor.grid_map.set_room_map(room_map)
            floor.grid_map.set_cat_map(cat_map)

    def save(self, output_path: str):
        """
        Save the building data to a yaml file and the floor and category maps to png files.
        To read the map, use the following code:
            room_map = cv2.imread(room_map_path, cv2.IMREAD_GRAYSCALE).astype(np.int32) + SOC.NA
            cat_map = cv2.imread(cat_map_path, cv2.IMREAD_GRAYSCALE).astype(np.int32) + SOC.NA
        :param output_path:
        :return:
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        assert os.path.isdir(output_path), f"{output_path} is not a directory"
        building_dict = self.convert_to_dict()
        yaml_file = os.path.join(output_path, f"building.yaml")
        with open(yaml_file, "w") as file:
            yaml.dump(building_dict, file)

        room_maps_dir = os.path.join(output_path, "room_maps")
        if not os.path.exists(room_maps_dir):
            os.makedirs(room_maps_dir)
        assert os.path.isdir(room_maps_dir), f"{room_maps_dir} is not a directory"

        cat_maps_dir = os.path.join(output_path, "cat_maps")
        if not os.path.exists(cat_maps_dir):
            os.makedirs(cat_maps_dir)
        assert os.path.isdir(cat_maps_dir), f"{cat_maps_dir} is not a directory"

        for floor_num, floor in self.floors.items():
            room_map_path = os.path.join(room_maps_dir, f"{floor_num}.png")
            cat_map_path = os.path.join(cat_maps_dir, f"{floor_num}.png")
            # png file does not support negative values, so we add -SOC.NA to all values
            cv2.imwrite(room_map_path, (floor.grid_map.get_room_map() - SOC.NA).astype(np.uint8))
            cv2.imwrite(cat_map_path, (floor.grid_map.get_cat_map() - SOC.NA).astype(np.uint8))

    def as_nx_graph(self):
        graph = nx.Graph()
        building_label = f"Building\n{self.name}"
        graph.add_node(building_label)
        nodes = [building_label]
        building_nodes = [building_label]
        floor_nodes = []
        room_nodes = []
        obj_nodes = []
        for floor_num, floor in self.floors.items():
            floor_label = f"Floor {floor_num}"
            nodes.append(floor_label)
            floor_nodes.append(floor_label)
            graph.add_node(floor_label)
            graph.add_edge(building_label, floor_label)
            for room_id, room in floor.rooms.items():
                room_label = f"Room {room.uuid}\n{room.scene_category}"
                nodes.append(room_label)
                room_nodes.append(room_label)
                graph.add_node(room_label)
                graph.add_edge(floor_label, room_label)
                for obj_id, obj in room.objects.items():
                    obj_label = f"Obj. {obj.uuid}\n{obj.class_}"
                    nodes.append(obj_label)
                    obj_nodes.append(obj_label)
                    graph.add_node(obj_label)
                    graph.add_edge(room_label, obj_label)
        return graph, nodes, building_nodes, floor_nodes, room_nodes, obj_nodes

    def visualize_as_graph(self, output_path: str, show: bool = False):
        graph, nodes, building_nodes, floor_nodes, room_nodes, obj_nodes = self.as_nx_graph()
        graph_fig = plt.figure(figsize=(2560 / 300, 1440 / 300), dpi=300)
        graph_ax = graph_fig.add_subplot(111)
        node_colors = []
        for node in nodes:
            if node in building_nodes:
                node_colors.append("#1f78b4")
            elif node in floor_nodes:
                node_colors.append("#33a02c")
            elif node in room_nodes:
                node_colors.append("#e31a1c")
            elif node in obj_nodes:
                node_colors.append("#ff7f00")
        nx.draw(
            graph,
            ax=graph_ax,
            with_labels=True,
            font_size=4,
            node_size=400,
            pos=nx.nx_agraph.graphviz_layout(graph),  # best
            # pos=nx.nx_pydot.pydot_layout(graph),  # good
            # pos=nx.kamada_kawai_layout(graph),  # good
            # pos=nx.drawing.arf_layout(graph),  # bad
            # pos=nx.shell_layout(graph),  # bad
            # pos=nx.random_layout(graph),  # bad
            # pos=nx.planar_layout(graph),  # bad
            # pos=nx.circular_layout(graph),  # bad
            # pos=nx.spectral_layout(graph),  # bad
            # pos=nx.spiral_layout(graph),  # bad
            # pos=nx.spring_layout(graph),  # fair
            node_color=node_colors,
        )
        plt.tight_layout()
        plt.savefig(output_path)
        if show:
            plt.pause(0.1)
