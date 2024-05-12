from typing import List

import numpy as np


class Base:
    count = 0

    def __init__(self):
        Base.count += 1
        self.uuid = Base.count
        self.parent_uuid: int = -1  # parent node uuid

    def convert_to_dict(self):
        return {"uuid": self.uuid, "parent_uuid": self.parent_uuid}


class Object(Base):
    def __init__(self):
        super().__init__()
        # Object 3D Scene Graph attributes
        self.action_affordance: List[str] = []  # list of possible actions
        self.floor_area: float = 0  # 2D floor area in sq.meters
        self.surface_coverage: float = 0  # total surface coverage in sq.meters
        self.class_: str = ""  # object label
        self.id: int = -1  # unique object id per building
        self.location: np.ndarray = np.zeros(3)  # 3D coordinates of object center's location
        self.material: List[str] = []  # list of main object materials
        self.size: np.ndarray = np.zeros(3)  # 3D Size of object
        self.volume: float = 0  # 3D volume of object computed from 3D convex hull (cubic meters)
        self.parent_room: int = -1  # parent room that contains this object (room id)
        self.grid_map_min: np.ndarray = np.zeros(2, dtype=int)  # min x, y coordinates of object in grid map
        self.grid_map_max: np.ndarray = np.zeros(2, dtype=int)  # max x, y coordinates of object in grid map

    def set_attribute(self, attribute, value):
        # Set an object attribute
        if attribute not in self.__dict__.keys():
            print("Unknown object attribute: {}".format(attribute))
            return
        if attribute == "id":
            self.__dict__[attribute] = int(value)
        elif attribute == "surface_coverage":
            self.__dict__[attribute] = value.item()
        else:
            self.__dict__[attribute] = value

    def get_attribute(self, attribute):
        # Get an object attribute
        if attribute not in self.__dict__.keys():
            print("Unknown object attribute: {}".format(attribute))
            return -1
        return self.__dict__[attribute]

    def convert_to_dict(self):
        ret_dict = super().convert_to_dict()
        ret_dict["type"] = "kObject"
        for key, value in self.__dict__.items():
            if key in ["size", "location", "grid_map_min", "grid_map_max"]:
                ret_dict[key] = value.tolist()
            elif key == "class_":
                ret_dict["name"] = value
            elif key == "parent_room":
                ret_dict["parent_id"] = value
            else:
                ret_dict[key] = value
        return ret_dict

    def load_from_dict(self, obj_dict: dict):
        assert obj_dict["type"] == "kObject", "Not an object dict!"
        for key, value in obj_dict.items():
            if key in ["size", "location", "grid_map_min", "grid_map_max"]:
                self.__dict__[key] = np.array(value)
            elif key == "name":
                self.__dict__["class_"] = value
            elif key == "parent_id":
                self.__dict__["parent_room"] = value
            else:
                self.__dict__[key] = value
