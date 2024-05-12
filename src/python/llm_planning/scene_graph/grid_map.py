import numpy as np

from llm_planning.scene_graph.special_object_categories import SOC


class GridMap:
    def __init__(self, cell_size, origin, map_size):
        self._origin = origin[:2]
        self._cell_size = cell_size[:2]
        self._cat_map = np.ones((map_size[0], map_size[1])) * SOC.NA
        self._room_map = np.ones((map_size[0], map_size[1])) * SOC.NA

    def xy_to_rc(self, xy_coords):
        if len(xy_coords.shape) == 1:
            x_coords = xy_coords[0]
            y_coords = xy_coords[1]

            # r = np.round((self._origin[1] - y_coords) / self._cell_size[1])
            # c = np.round((x_coords - self._origin[0]) / self._cell_size[0])
            # rc = np.array([r, c])
            r = np.floor((x_coords - self._origin[0]) / self._cell_size[0])
            c = np.floor((y_coords - self._origin[1]) / self._cell_size[1])
            rc = np.array([r, c])
        else:
            x_coords = xy_coords[:, 0]
            y_coords = xy_coords[:, 1]

            # r = np.round((self._origin[1] - y_coords) / self._cell_size[1])[:, None]
            # c = np.round((x_coords - self._origin[0]) / self._cell_size[0])[:, None]
            # rc = np.hstack((r, c))
            r = np.floor((x_coords - self._origin[0]) / self._cell_size[0])[:, None]
            c = np.floor((y_coords - self._origin[1]) / self._cell_size[1])[:, None]
            rc = np.hstack((r, c))

        return rc.astype(int)

    def rc_to_xy(self, rc_coords):
        if len(rc_coords.shape) == 1:
            r_coords = rc_coords[0]
            c_coords = rc_coords[1]

            x = self._origin[0] + (c_coords + 0.5) * self._cell_size[0]
            y = self._origin[1] + (r_coords + 0.5) * self._cell_size[1]
            xy = np.array([x, y])
        else:
            r_coords = rc_coords[:, 0]
            c_coords = rc_coords[:, 1]

            x = self._origin[0] + (c_coords + 0.5) * self._cell_size[0]
            y = self._origin[1] + (r_coords + 0.5) * self._cell_size[1]
            xy = np.hstack((x[:, None], y[:, None]))
        return xy

    def set_cat_element(self, r, c, value):
        self._cat_map[r, c] = value

    def set_room_element(self, r, c, value):
        self._room_map[r, c] = value

    def set_cat_map(self, cat_map):
        assert cat_map.shape == self._cat_map.shape, "Map shapes need to be the same!"
        self._cat_map = cat_map.copy()

    def set_room_map(self, room_map):
        assert room_map.shape == self._room_map.shape, "Map shapes need to be the same!"
        self._room_map = room_map.copy()

    def get_cat_map(self):
        return self._cat_map.copy()

    def get_room_map(self):
        return self._room_map.copy()

    def get_map_size(self):
        return self._cat_map.shape

    def copy(self):
        copy_map = GridMap(cell_size=self._cell_size, origin=self._origin, map_size=self._cat_map.shape)
        copy_map.set_cat_map(self._cat_map)
        copy_map.set_room_map(self._room_map)
        return copy_map

    def convert_to_dict(self):
        ret_dict = {
            "origin": self._origin.tolist(),
            "cell_size": self._cell_size.tolist(),
            "map_size": list(self._cat_map.shape),
        }
        return ret_dict
