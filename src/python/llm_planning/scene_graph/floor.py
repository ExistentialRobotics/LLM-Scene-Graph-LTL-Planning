from typing import Dict
from typing import Optional

import numpy as np

from llm_planning.scene_graph.grid_map import GridMap
from llm_planning.scene_graph.room import Base, Room
from llm_planning.scene_graph.visualize import draw_cat_map
from llm_planning.scene_graph.visualize import draw_room_map


class Floor(Base):
    def __init__(self, parent_building: int = -1, floor_num: int = -1, grid_map: GridMap = None):
        super().__init__()
        self.parent_building: int = parent_building  # parent building that contains this floor
        self.floor_num: int = floor_num  # floor number
        self.rooms: Dict[int, Room] = {}  # room id to room object
        self.ground_z: float = 0  # ground z coordinate
        self.grid_map: GridMap = grid_map
        self.down_stairs_id: int = -1
        self.up_stairs_id: int = -1
        self.down_stairs_uuid: int = -1
        self.up_stairs_uuid: int = -1
        self.down_stairs_portal: Optional[np.ndarray] = None
        self.up_stairs_portal: Optional[np.ndarray] = None
        self.up_stairs_cost: float = float("inf")
        self.down_stairs_cost: float = float("inf")

    def convert_to_dict(self):
        ret_dict = super().convert_to_dict()
        ret_dict["type"] = "kFloor"
        for key, value in self.__dict__.items():
            if key == "rooms":
                rooms = {}
                for room_id, room in self.rooms.items():
                    rooms[room_id] = room.convert_to_dict()
                ret_dict[key] = rooms
            elif key == "grid_map":
                grid_map_dict = value.convert_to_dict()
                ret_dict["grid_map_origin"] = grid_map_dict["origin"]
                ret_dict["grid_map_resolution"] = grid_map_dict["cell_size"]
                ret_dict["grid_map_size"] = grid_map_dict["map_size"]
            elif key == "floor_num":
                ret_dict["id"] = value
            elif key == "parent_building":
                ret_dict["parent_id"] = value
            elif key in ["down_stairs_portal", "up_stairs_portal"] and value is not None:
                ret_dict[key] = value.tolist()
            else:
                ret_dict[key] = value
        ret_dict["name"] = chr(ord("A") + self.floor_num)
        ret_dict["room_map"] = f"room_maps/{self.floor_num}.png"
        ret_dict["cat_map"] = f"cat_maps/{self.floor_num}.png"
        ret_dict["num_rooms"] = len(self.rooms)
        return ret_dict

    def load_from_dict(self, floor_dict: dict):
        assert floor_dict["type"] == "kFloor", "Not a floor dict!"
        grid_map_dict = dict()
        for key, value in floor_dict.items():
            if key == "rooms":
                for room_id, room_dict in value.items():
                    room = Room()
                    room.load_from_dict(room_dict)
                    self.rooms[room_id] = room
            elif key == "id":
                self.floor_num = value
            elif key == "parent_id":
                self.parent_building = value
            elif key == "grid_map_origin":
                grid_map_dict["origin"] = np.array(value)
            elif key == "grid_map_resolution":
                grid_map_dict["cell_size"] = np.array(value)
            elif key == "grid_map_size":
                grid_map_dict["map_size"] = np.array(value)
            elif key in ["down_stairs_portal", "up_stairs_portal"] and value is not None:
                self.__dict__[key] = np.array(value)
            elif key in ["room_map", "cat_map"]:
                continue
            else:
                self.__dict__[key] = value
        self.grid_map = GridMap(**grid_map_dict)

    def draw_room_map(self, output_path: str = None, hold: bool = False):
        room_map = self.grid_map.get_room_map()
        room_categories = dict()
        room_connections = []
        for room_id, room in self.rooms.items():
            if room.scene_category == "staircase":
                continue
            room_categories[room_id] = room.scene_category
            if room_id == self.up_stairs_id:
                loc1 = np.argwhere(room_map == room_id).mean(axis=0).astype(int)
            else:
                loc1 = self.grid_map.xy_to_rc(room.location[:2])
            for connected_room_id in room.connected_room_ids:
                if connected_room_id == self.down_stairs_id or connected_room_id == self.up_stairs_id:
                    loc2 = np.argwhere(room_map == connected_room_id).mean(axis=0).astype(int)
                else:
                    loc2 = self.grid_map.xy_to_rc(self.rooms[connected_room_id].location[:2])
                room_connections.append((loc1, loc2))
        room_uuids = {room_id: room.uuid for room_id, room in self.rooms.items()}
        if self.down_stairs_id > 0:
            room_categories[self.down_stairs_id] = "down_stairs"
            room_uuids[self.down_stairs_id] = self.down_stairs_uuid
        if self.up_stairs_id > 0:
            room_categories[self.up_stairs_id] = "up_stairs"
            room_uuids[self.up_stairs_id] = self.up_stairs_uuid
        return draw_room_map(
            room_map,
            room_categories,
            room_uuids,
            room_connections,
            {room_id: room.door_grids for room_id, room in self.rooms.items()},
            title=f"Room Map of Floor {self.floor_num}",
            output_path=output_path,
            show=True,
            hold=hold,
        )

    def draw_cat_map(self, output_path: str = None, hold: bool = False):
        cat_map = self.grid_map.get_cat_map()
        obj_categories = dict()
        obj_uuids = dict()
        for room_id, room in self.rooms.items():
            for obj_id, obj in room.objects.items():
                obj_categories[obj_id] = obj.class_
                obj_uuids[obj_id] = obj.uuid
        return draw_cat_map(
            cat_map,
            obj_categories,
            obj_uuids,
            self.up_stairs_portal,
            self.down_stairs_portal,
            {room_id: room.door_grids for room_id, room in self.rooms.items()},
            title=f"Object Category Map of Floor {self.floor_num}",
            output_path=output_path,
            show=True,
            hold=hold,
        )
