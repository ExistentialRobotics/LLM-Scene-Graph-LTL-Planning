from typing import Dict, Set, List, Tuple

import numpy as np

from llm_planning.scene_graph.obj import Base, Object


class Room(Base):
    def __init__(self):
        super().__init__()
        # Room 3D Scene Graph attributes
        self.floor_area: float = 0  # 2D floor area in sq.meters
        self.floor_number: int = -1  # index of floor that contains the space
        self.id: int = -1  # unique space id per building
        self.location: np.ndarray = np.zeros(3)  # 3D coordinates of room center's location
        self.scene_category: str = ""  # function of this room
        self.size: np.ndarray = np.zeros(3)  # 3D Size of room
        self.volume: float = 0  # 3D volume of room computed from 3D convex hull (cubic meters)
        self.parent_building: int = -1  # parent building that contains this room

        # downstream graph layer
        self.objects: Dict[int, Object] = {}

        # room connections
        self.connected_room_ids: Set[int] = set()  # needs to be sorted out
        self.connected_room_uuids: Set[int] = set()  # needs to be sorted out
        self.door_grids: Dict[int, List[Tuple[int, int]]] = dict()  # grids that are doors of connected rooms
        self.grid_map_min: np.ndarray = np.zeros(2, dtype=int)  # min x, y coordinates of room in grid map
        self.grid_map_max: np.ndarray = np.zeros(2, dtype=int)  # max x, y coordinates of room in grid map

    def set_attribute(self, attribute, value):
        # Set a room attribute
        if attribute not in self.__dict__.keys():
            print("Unknown room attribute: {}".format(attribute))
            return
        if attribute == "floor_number":
            self.__dict__[attribute] = ord(value) - ord("A")
        else:
            self.__dict__[attribute] = value

    def get_attribute(self, attribute):
        # Get a room attribute
        if attribute not in self.__dict__.keys():
            print("Unknown room attribute: {}".format(attribute))
            return -1
        return self.__dict__[attribute]

    def convert_to_dict(self):
        ret_dict = super().convert_to_dict()
        ret_dict["type"] = "kRoom"
        for key, value in self.__dict__.items():
            if key in ["size", "location", "grid_map_min", "grid_map_max"]:
                ret_dict[key] = value.tolist()
            elif key in ["connected_room_ids", "connected_room_uuids"]:
                ret_dict[key] = list(value)
            elif key == "floor_number":
                ret_dict["parent_id"] = value
            elif key == "scene_category":
                ret_dict["name"] = value
            elif key == "objects":
                objects = {}
                for object_key, obj in self.objects.items():
                    objects[object_key] = obj.convert_to_dict()
                ret_dict[key] = objects
            else:
                ret_dict[key] = self.__dict__[key]
        ret_dict["num_objects"] = len(self.objects)
        return ret_dict

    def load_from_dict(self, room_dict: dict):
        assert room_dict["type"] == "kRoom", "Not a room dict!"
        for key, value in room_dict.items():
            if key == "objects":
                for obj_key, obj_dict in value.items():
                    obj = Object()
                    obj.load_from_dict(obj_dict)
                    self.objects[obj_key] = obj
            elif key in ["connected_room_ids", "connected_room_uuids"]:
                self.__dict__[key] = set(value)
            elif key == "parent_id":
                self.floor_number = value
            elif key == "name":
                self.scene_category = value
            elif key in ["size", "location", "grid_map_min", "grid_map_max"]:
                self.__dict__[key] = np.array(value)
            else:
                self.__dict__[key] = value
