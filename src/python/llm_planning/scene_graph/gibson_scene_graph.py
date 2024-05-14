"""
This file converts the Gibson scene graph data to the format used by this project.
"""

import argparse
import os
import pickle

import cv2
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ray
import shapely
import trimesh
import vedo

from llm_planning.scene_graph.building import Building
from llm_planning.scene_graph.color_map import COLOR_MAP
from llm_planning.scene_graph.floor import Floor
from llm_planning.scene_graph.grid_map import GridMap
from llm_planning.scene_graph.obj import Object
from llm_planning.scene_graph.room import Room
from llm_planning.scene_graph.special_object_categories import SOC


def prune_unused_vertices(vertices, faces):
    used_vertices_inds = np.unique(faces.flatten())
    used_vertices_inds.sort()
    used_vertices_inds = used_vertices_inds.tolist()

    inds_mapping = np.repeat(-1, (vertices.shape[0],))
    inds_mapping.put(used_vertices_inds, np.arange(len(used_vertices_inds)))
    faces = inds_mapping[faces]
    faces = np.array(faces).reshape((-1, 3))
    vertices = vertices[used_vertices_inds]
    return vertices, faces


def get_biggest_polygon(polygons: trimesh.path.Path2D):
    if len(polygons.polygons_closed) == 0:
        return None
    polygon = polygons.polygons_closed[0]
    for p in polygons.polygons_closed[1:]:
        if p is None:
            continue
        if p.area > polygon.area:
            polygon = p
    return polygon


def get_contour_polygons(polygons: trimesh.path.Path2D):
    contour_polygons = []
    for i in polygons.root:
        contour_polygons.append(polygons.polygons_closed[i])
    return contour_polygons


def get_polygon_xy(polygon):
    if isinstance(polygon, shapely.MultiPolygon):
        return np.array(polygon.envelope.exterior.xy).T
    else:
        return np.array(polygon.exterior.xy).T


def floor_num_to_char(floor_num: int) -> str:
    return chr(ord("A") + floor_num)


def floor_char_to_num(floor_char: str) -> int:
    return ord(floor_char) - ord("A")


class BuildingLoader:
    def __init__(self, npz_file: str, mesh_file: str, cell_size: float = 0.05, debug: bool = False):
        self.debug = debug
        self.building = Building()
        self.npz_data = np.load(npz_file, allow_pickle=True)["output"].item()
        mesh_data = trimesh.load(mesh_file)
        # coordinate system of location data: x: out of screen, y: to the right, z: up
        # coordinate system of mesh data: x out of screen, y: up, z: to the left
        # rotate the mesh coordinate system clockwise by 90 degrees around x-axis to match the location data
        self.mesh_vertices = mesh_data.vertices[:, [0, 2, 1]]  # (x, z, -y) to (x, -y, z)
        self.mesh_vertices[:, 1] *= -1  # (x, -y, z) to (x, y, z)
        self.mesh_faces = mesh_data.faces
        self.mesh_faces_pos = np.mean(self.mesh_vertices[mesh_data.faces, :], axis=1)
        self.metric_min = self.mesh_vertices.min(axis=0) - 0.2  # (x, y, z)
        self.metric_max = self.mesh_vertices.max(axis=0) + 0.2  # (x, y, z)
        # row: x (down), col: y (right)
        self.map_size = np.round((self.metric_max - self.metric_min)[:2] / cell_size).astype(int)
        if self.map_size[0] % 2 == 0:
            self.map_size[0] += 1
        if self.map_size[1] % 2 == 0:
            self.map_size[1] += 1

        self.cell_size = (self.metric_max - self.metric_min)[:2] / self.map_size  # (x_size, y_size)
        self.env_name = os.path.splitext(os.path.basename(npz_file))[0].split("_")[1]

        # set environment-specific parameters
        self.ground_height = dict(Merom=0.4)
        self.wall_z_min = dict(Hiteman=0.5, Merom=0.8)
        self.wall_z_max = dict(Hiteman=0.8, Merom=1.4)
        self.floor_polygon_voxel_size = dict(Hiteman=0.3)
        self.floor_polygon_apad = dict()
        self.floor_polygon_buffer = dict()
        self.wall_polygon_voxel_size = dict(Hiteman=0.03)
        self.wall_polygon_apad = dict(Hiteman=0.1)
        self.wall_polygon_buffer = dict(Allensville=0.1, Hiteman=0.05)
        self.up_stairs_ws = dict(Beechwood=9, Hiteman=5)
        self.down_stairs_ws = dict(Beechwood=9, Hiteman=5, Merom=5)
        self.up_stairs_buffer = dict()
        self.down_stairs_buffer = dict(Beechwood=-0.2, Hiteman=0.2, Merom=0.0)
        self.room_polygon_voxel_size1 = dict(Hanson=0.05, Merom=0.02)
        self.room_polygon_apad1 = dict(Merom=-0.06)
        self.room_polygon_voxel_size2 = dict(Hanson=0.07)
        self.room_polygon_apad2 = dict()
        self.room_connection_dilate_iter = dict(Collierville=2)

        self.load_scene_graph_file()
        self.load_mesh_file()

    @property
    def npz_building_data(self) -> dict:
        return self.npz_data["building"]

    @property
    def npz_room_data(self) -> dict:
        return self.npz_data["room"]

    @property
    def npz_object_data(self) -> dict:
        return self.npz_data["object"]

    @property
    def room_inst_segmentation(self) -> np.ndarray:
        return self.npz_building_data["room_inst_segmentation"]

    @property
    def object_inst_segmentation(self) -> np.ndarray:
        return self.npz_building_data["object_inst_segmentation"]

    def get_floor(self, floor_num: int) -> Floor:
        if floor_num not in self.building.floors.keys():
            floor = Floor(
                self.building.id,
                floor_num,
                GridMap(
                    cell_size=self.cell_size,
                    origin=self.metric_min[:2],
                    map_size=self.map_size,
                ),
            )
            floor.parent_uuid = self.building.uuid
            self.building.floors[floor_num] = floor
        return self.building.floors[floor_num]

    def load_scene_graph_file(self):
        print("Loading scene graph file...")
        # set building attributes
        for key, value in self.npz_building_data.items():
            if key in [
                "object_inst_segmentation",
                "room_inst_segmentation",
                "object_voxel_occupancy",
                "room_voxel_occupancy",
                "gibson_split",
                "num_cameras",
                "num_objects",
                "voxel_size",
                "voxel_centers",
                "voxel_resolution",
            ]:
                continue
            self.building.set_attribute(key, value)

        # set room attributes
        for room_id, room_data in self.npz_room_data.items():
            room_id = int(room_id)
            room = Room()
            for key, value in room_data.items():
                if key in ["inst_segmentation", "voxel_occupancy"]:
                    continue
                room.set_attribute(key, value)
            floor = self.get_floor(room.floor_number)
            room.parent_uuid = floor.uuid
            floor.rooms[room_id] = room
            self.building.room_id_to_floor_num[room_id] = room.floor_number

        # set object attributes
        for obj_id, obj_data in self.npz_object_data.items():
            obj_id = int(obj_id)
            parent_room = int(obj_data["parent_room"])
            floor_num = self.building.room_id_to_floor_num[parent_room]
            obj = Object()
            for key, value in obj_data.items():
                if key in ["inst_segmentation", "voxel_occupancy", "tactile_texture", "visual_texture"]:
                    continue
                obj.set_attribute(key, value)
            obj.parent_uuid = self.building.floors[floor_num].rooms[parent_room].uuid
            self.building.floors[floor_num].rooms[parent_room].objects[obj_id] = obj
            self.building.object_id_to_room_id[obj_id] = parent_room

    @ray.remote
    def extract_room_planform_polygon(self, room_id: int):
        vertices = self.mesh_vertices
        faces = self.mesh_faces

        face_room_cat = self.room_inst_segmentation[:, 0].astype(int)
        room_mask = face_room_cat == room_id
        room_vertices, room_faces = prune_unused_vertices(vertices, faces[room_mask])

        mesh_org = trimesh.Trimesh(vertices=room_vertices, faces=room_faces)
        t = room_vertices.min(axis=0)

        voxel_size1 = self.room_polygon_voxel_size1.get(self.env_name, 0.03)
        transform = np.diag([voxel_size1, voxel_size1, voxel_size1, 1])
        transform[:3, 3] = t
        polygon1 = get_biggest_polygon(
            mesh_org.voxelized(pitch=voxel_size1, max_iter=20)
            .marching_cubes.apply_transform(transform)
            .projected(
                normal=np.array([0, 0, 1]),
                max_regions=20000,
                apad=self.room_polygon_apad1.get(self.env_name, 0.1),
                ignore_sign=True,
            )
        ).buffer(0.05)

        voxel_size2 = self.room_polygon_voxel_size2.get(self.env_name, 0.05)
        transform = np.diag([voxel_size2, voxel_size2, voxel_size2, 1])
        transform[:3, 3] = t
        polygon2 = get_biggest_polygon(
            mesh_org.voxelized(voxel_size2)
            .marching_cubes.apply_transform(transform)
            .projected(
                normal=np.array([0, 0, 1]),
                max_regions=20000,
                apad=self.room_polygon_apad2.get(self.env_name, 0.2),
                ignore_sign=True,
            )
        )

        if self.debug:
            import vedo

            vedo.show(
                vedo.Mesh([room_vertices, room_faces], c="yellow"),
                vedo.Line(get_polygon_xy(polygon1), c="red").z(room_vertices[:, 2].mean()),
                vedo.Line(get_polygon_xy(polygon2), c="green").z(room_vertices[:, 2].mean()),
                axes=1,
                interactive=True,
            )

        intersection = shapely.intersection(polygon1, polygon2)
        union = shapely.union(polygon1, polygon2)
        r = abs(intersection.area - union.area) / union.area
        if r > 0.4:
            if self.debug:
                print(f"ratio: {r}, warning: object planform polygon area difference is too large!")
            polygon = polygon1 if polygon1.area > polygon2.area else polygon2
            polygon = get_polygon_xy(polygon)
        elif r < 0.01:
            if self.debug:
                print(f"ratio: {r}, use union to get room planform polygon")
            polygon = get_polygon_xy(union)  # (N, 2)
        else:
            if self.debug:
                print(f"ratio: {r}, use intersection to get room planform polygon")
            polygon = get_polygon_xy(intersection)  # (N, 2)

        if self.debug:
            import vedo

            vedo.show(
                vedo.Mesh([room_vertices, room_faces], c="yellow"),
                vedo.Line(polygon[:, 0], polygon[:, 1], c="red").z(room_vertices[:, 2].mean()),
                axes=1,
                interactive=True,
            )

        return polygon

    @ray.remote
    def extract_object_planform_polygon(self, obj_id: int):
        vertices = self.mesh_vertices
        faces = self.mesh_faces

        face_obj_cat = self.object_inst_segmentation[:, 0].astype(int)
        obj_mask = face_obj_cat == obj_id
        obj_vertices_inds = np.unique(faces[obj_mask].flatten())
        obj_vertices_inds.sort()
        obj_vertices_inds = obj_vertices_inds.tolist()
        obj_vertices = vertices[obj_vertices_inds]
        obj_faces = faces[obj_mask].flatten().tolist()
        inds_mapping = np.repeat(-1, (vertices.shape[0],))
        inds_mapping[obj_vertices_inds] = np.arange(len(obj_vertices_inds))
        obj_faces = inds_mapping[obj_faces]
        obj_faces = np.array(obj_faces).reshape((-1, 3))

        mesh_org = trimesh.Trimesh(vertices=obj_vertices, faces=obj_faces)
        t = obj_vertices.min(axis=0)

        voxel_size1 = 0.04
        transform = np.diag([voxel_size1, voxel_size1, voxel_size1, 1])
        transform[:3, 3] = t
        polygon1 = get_biggest_polygon(
            mesh_org.voxelized(pitch=voxel_size1)
            .marching_cubes.apply_transform(transform)
            .projected(
                normal=np.array([0, 0, 1]),
                max_regions=20000,
                apad=0.15,
                ignore_sign=True,
            )
        )

        voxel_size2 = 0.06
        transform = np.diag([voxel_size2, voxel_size2, voxel_size2, 1])
        transform[:3, 3] = t
        polygon2 = get_biggest_polygon(
            mesh_org.voxelized(voxel_size2)
            .marching_cubes.apply_transform(transform)
            .projected(
                normal=np.array([0, 0, 1]),
                max_regions=20000,
                apad=0.2,
                ignore_sign=True,
            )
        )

        if self.debug:
            import vedo

            vedo.show(
                vedo.Mesh([obj_vertices, obj_faces], c="blue"),
                vedo.Line(get_polygon_xy(polygon1), c="red").z(obj_vertices[:, 2].mean()),
                vedo.Line(get_polygon_xy(polygon2), c="green").z(obj_vertices[:, 2].mean()),
                axes=1,
                interactive=True,
            )

        if polygon2 is None:
            polygon = get_polygon_xy(polygon1)
            return polygon, obj_vertices[:, 2].mean()

        intersection = shapely.intersection(polygon1, polygon2)
        union = shapely.union(polygon1, polygon2)
        r = abs(intersection.area - union.area) / union.area
        if r > 0.4:
            if self.debug:
                print(f"ratio: {r}, warning: object planform polygon area difference is too large!")
            polygon = polygon1 if polygon1.area > polygon2.area else polygon2
            polygon = get_polygon_xy(polygon)
        elif r < 0.01:
            if self.debug:
                print(f"ratio: {r}, use union to get room planform polygon")
            polygon = get_polygon_xy(union)  # (N, 2)
        else:
            if self.debug:
                print(f"ratio: {r}, use intersection to get room planform polygon")
            polygon = get_polygon_xy(intersection)  # (N, 2)

        return polygon, obj_vertices[:, 2].mean()

    def extract_floor_ground(self, floor: Floor):
        print("Extracting floor ground...")
        vertices = self.mesh_vertices
        faces = self.mesh_faces

        down_stairs_mask = None
        down_stairs_room_id = -1
        up_stairs_mask = None
        up_stairs_room_id = -1

        face_room_cat = self.room_inst_segmentation[:, 0].astype(int)

        down_floor_num = floor.floor_num - 1
        if down_floor_num in self.building.floors:
            for room in self.building.floors[down_floor_num].rooms.values():
                if room.scene_category == "staircase":
                    down_stairs_mask = face_room_cat == room.id  # down-stairs are always on the previous floor
                    down_stairs_room_id = room.id
                    break

        floor_mask = np.zeros_like(face_room_cat)
        for room in floor.rooms.values():
            floor_mask = np.logical_or(floor_mask, face_room_cat == int(room.id))
            if room.scene_category == "staircase":
                up_stairs_mask = face_room_cat == room.id
                up_stairs_room_id = room.id

        floor_mask = np.logical_and(floor_mask, self.object_inst_segmentation[:, 0].astype(int) == 0)
        floor_vertices, floor_faces = prune_unused_vertices(vertices, faces[floor_mask])

        floor_faces_pos = np.mean(floor_vertices[floor_faces], axis=1)
        z = floor_faces_pos[:, 2]
        shell_z_min = z.min()
        ground_z_max = shell_z_min + self.ground_height.get(self.env_name, 0.3)
        ground_mask = np.logical_and(shell_z_min <= z, z <= ground_z_max)
        wall_mask = np.logical_and(
            ground_z_max + self.wall_z_min.get(self.env_name, 0.6) < z,
            z <= ground_z_max + self.wall_z_max.get(self.env_name, 1.3),
        )
        ground_vertices, ground_faces = prune_unused_vertices(floor_vertices, floor_faces[ground_mask])
        wall_vertices, wall_faces = prune_unused_vertices(floor_vertices, floor_faces[wall_mask])
        ground_z = ground_vertices[:, 2].mean().item()

        voxel_size = self.floor_polygon_voxel_size.get(self.env_name, 0.2)
        transform = np.diag([voxel_size, voxel_size, voxel_size, 1])
        transform[:3, 3] = ground_vertices.min(axis=0)
        floor_polygon = get_biggest_polygon(
            trimesh.Trimesh(vertices=ground_vertices, faces=ground_faces)
            .voxelized(pitch=voxel_size)
            .marching_cubes.apply_transform(transform)
            .projected(
                normal=np.array([0, 0, 1]),
                max_regions=20000,
                apad=self.floor_polygon_apad.get(self.env_name, 0.02),
                ignore_sign=True,
            )
        ).buffer(self.floor_polygon_buffer.get(self.env_name, -0.04))
        floor_polygon = get_polygon_xy(floor_polygon)  # (N, 2)

        voxel_size = self.wall_polygon_voxel_size.get(self.env_name, 0.14)
        transform = np.diag([voxel_size, voxel_size, voxel_size, 1])
        transform[:3, 3] = wall_vertices.min(axis=0)
        wall_polygons = (
            trimesh.Trimesh(vertices=wall_vertices, faces=wall_faces)
            .voxelized(voxel_size)
            .marching_cubes.apply_transform(transform)
            .projected(
                normal=np.array([0, 0, 1]),
                max_regions=20000,
                apad=self.wall_polygon_apad.get(self.env_name, 0.01),
                ignore_sign=True,
            )
            .polygons_closed
        )
        wall_polygons = sorted(wall_polygons, key=lambda x: x.area, reverse=True)  # descending order
        wall_polygons = [polygon for polygon in wall_polygons if polygon.area > 0.1]

        if self.debug:
            import vedo

            vedo.show(
                vedo.Mesh([ground_vertices, ground_faces], c="blue"),
                vedo.Mesh([wall_vertices, wall_faces], c="yellow"),
                *[
                    vedo.Line(get_polygon_xy(polygon), c="red").z(wall_vertices[:, 2].mean())
                    for polygon in wall_polygons
                ],
                axes=1,
                interactive=True,
            )

        down_stairs_polygon = None
        if down_stairs_mask is not None:
            down_stairs_vertices, down_stairs_faces = prune_unused_vertices(vertices, faces[down_stairs_mask])
            down_stairs_faces_z = np.mean(down_stairs_vertices[down_stairs_faces], axis=1)[:, 2]
            z_min = down_stairs_faces_z.min()
            z_max = down_stairs_faces_z.max()
            z_range = z_max - z_min
            down_stairs_mask = np.logical_and(z_max - z_range / 6 <= down_stairs_faces_z, down_stairs_faces_z <= z_max)
            down_stairs_vertices, down_stairs_faces = prune_unused_vertices(
                down_stairs_vertices, down_stairs_faces[down_stairs_mask]
            )
            voxel_size = 0.015
            transform = np.diag([voxel_size, voxel_size, voxel_size, 1])
            transform[:3, 3] = down_stairs_vertices.min(axis=0)
            down_stairs_polygon = get_biggest_polygon(
                trimesh.Trimesh(vertices=down_stairs_vertices, faces=down_stairs_faces)
                .voxelized(pitch=voxel_size)
                .marching_cubes.apply_transform(transform)
                .projected(normal=np.array([0, 0, 1]), max_regions=20000, apad=-0.01, ignore_sign=True)
            )

        up_stairs_polygon = None
        if up_stairs_mask is not None:
            up_stairs_vertices, up_stairs_faces = prune_unused_vertices(vertices, faces[up_stairs_mask])
            up_stairs_faces_z = np.mean(up_stairs_vertices[up_stairs_faces], axis=1)[:, 2]
            z_min = up_stairs_faces_z.min()
            z_max = up_stairs_faces_z.max()
            z_range = z_max - z_min
            up_stairs_mask = np.logical_and(z_min <= up_stairs_faces_z, up_stairs_faces_z <= z_min + z_range / 4)
            up_stairs_vertices, up_stairs_faces = prune_unused_vertices(
                up_stairs_vertices, up_stairs_faces[up_stairs_mask]
            )
            voxel_size = 0.015
            transform = np.diag([voxel_size, voxel_size, voxel_size, 1])
            transform[:3, 3] = up_stairs_vertices.min(axis=0)
            up_stairs_polygon = get_biggest_polygon(
                trimesh.Trimesh(vertices=up_stairs_vertices, faces=up_stairs_faces)
                .voxelized(pitch=voxel_size)
                .marching_cubes.apply_transform(transform)
                .projected(normal=np.array([0, 0, 1]), max_regions=20000, apad=-0.01, ignore_sign=True)
            )

        return (
            floor_polygon,
            wall_polygons,
            ground_z,
            down_stairs_room_id,
            down_stairs_polygon,
            up_stairs_room_id,
            up_stairs_polygon,
        )

    def find_room_connections(
            self,
            room_map: np.ndarray,
            cat_map: np.ndarray,
            up_stairs_room_id: int = -1,
            down_stairs_room_id: int = -1,
    ):
        room_connections = dict()
        room_door_grids = dict()
        room_map = room_map.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ground_mask: np.ndarray = cat_map != SOC.GROUND
        if up_stairs_room_id > 0:
            up_stairs_mask = room_map == up_stairs_room_id
            ground_mask[up_stairs_mask] = False
        if down_stairs_room_id > 0:
            down_stairs_mask = room_map == down_stairs_room_id
            ground_mask[down_stairs_mask] = False
        ground_mask = cv2.dilate(
            ground_mask.astype(np.uint8),
            kernel=kernel,
            iterations=self.room_connection_dilate_iter.get(self.env_name, 15),
        ).astype(bool)
        dx = room_map[:, 1:] - room_map[:, :-1]
        indices = np.argwhere(dx != 0)
        for ind in indices:
            r = int(ind[0])
            c1 = int(ind[1])
            c2 = int(ind[1]) + 1
            room1 = int(room_map[r, c1])
            room2 = int(room_map[r, c2])
            if room1 <= 0 or room2 <= 0:
                continue
            is_ground = not ground_mask[r, c1] and not ground_mask[r, c2]
            if room1 not in room_connections.keys():
                if is_ground:
                    room_connections[room1] = set()
                room_door_grids[room1] = dict()
            if room2 not in room_connections.keys():
                if is_ground:
                    room_connections[room2] = set()
                room_door_grids[room2] = dict()
            if is_ground:
                room_connections[room1].add(room2)
                room_connections[room2].add(room1)
            if cat_map[r, c1] == SOC.GROUND:  # ground pixel in room1, add it to room2's door grid
                if room1 not in room_door_grids[room2]:
                    room_door_grids[room2][room1] = []
                room_door_grids[room2][room1].append((r, c1))
            if cat_map[r, c2] == SOC.GROUND:  # ground pixel in room2, add it to room1's door grid
                if room2 not in room_door_grids[room1]:
                    room_door_grids[room1][room2] = []
                room_door_grids[room1][room2].append((r, c2))
        dy = room_map[1:, :] - room_map[:-1, :]
        indices = np.argwhere(dy != 0)
        for ind in indices:
            r1 = int(ind[0])
            r2 = int(ind[0]) + 1
            c = int(ind[1])
            room1 = int(room_map[r1, c])
            room2 = int(room_map[r2, c])
            if room1 <= 0 or room2 <= 0:
                continue
            is_ground = not ground_mask[r1, c] and not ground_mask[r2, c]
            if room1 not in room_connections.keys():
                if is_ground:
                    room_connections[room1] = set()
                room_door_grids[room1] = dict()
            if room2 not in room_connections.keys():
                if is_ground:
                    room_connections[room2] = set()
                room_door_grids[room2] = dict()
            if is_ground:
                room_connections[room1].add(room2)
                room_connections[room2].add(room1)
            if cat_map[r1, c] == SOC.GROUND:  # ground pixel in room1, add it to room2's door grid
                if room1 not in room_door_grids[room2]:
                    room_door_grids[room2][room1] = []
                room_door_grids[room2][room1].append((r1, c))
            if cat_map[r2, c] == SOC.GROUND:  # ground pixel in room2, add it to room1's door grid
                if room2 not in room_door_grids[room1]:
                    room_door_grids[room1][room2] = []
                room_door_grids[room1][room2].append((r2, c))

        for room1, room2s in room_door_grids.items():
            for room2, door_grids in room2s.items():
                room_door_grids[room1][room2] = [list(x) for x in set(door_grids)]

        # extract staircase portals
        if up_stairs_room_id > 0:
            up_stairs_portal = []
            for _, door_grids in room_door_grids[up_stairs_room_id].items():
                up_stairs_portal += door_grids
            if len(up_stairs_portal) > 0:
                ws = self.up_stairs_ws.get(self.env_name, 4)
                tmp = np.array(up_stairs_portal)
                r_grids, c_grids = np.meshgrid(np.arange(-ws, ws + 1), np.arange(-ws, ws + 1))
                r_grids = r_grids.flatten()
                c_grids = c_grids.flatten()
                tmp = tmp[:, None, :] + np.array([r_grids, c_grids]).T[None, :, :]
                n, m = tmp.shape[:2]
                tmp = tmp.reshape((-1, 2))
                tmp = cat_map[tmp[:, 0], tmp[:, 1]].reshape(n, m)
                tmp = np.all(np.logical_or(tmp == SOC.STAIRS_UP, tmp == SOC.GROUND), axis=1)
                up_stairs_portal = np.array(up_stairs_portal)[tmp].mean(axis=0).astype(int)
                assert cat_map[up_stairs_portal[0], up_stairs_portal[1]] == SOC.GROUND
            else:
                up_stairs_portal = None
        else:
            up_stairs_portal = None
        if down_stairs_room_id > 0:
            down_stairs_portal = []
            for _, door_grids in room_door_grids[down_stairs_room_id].items():
                down_stairs_portal += door_grids
            if len(down_stairs_portal) > 0:
                ws = self.down_stairs_ws.get(self.env_name, 4)
                tmp = np.array(down_stairs_portal).astype(int)
                r_grids, c_grids = np.meshgrid(np.arange(-ws, ws + 1), np.arange(-ws, ws + 1))
                r_grids = r_grids.flatten()
                c_grids = c_grids.flatten()
                tmp = tmp[:, None, :] + np.array([r_grids, c_grids]).T[None, :, :]
                n, m = tmp.shape[:2]
                tmp = tmp.reshape((-1, 2))
                tmp = cat_map[tmp[:, 0], tmp[:, 1]].reshape(n, m)
                tmp = np.all(np.logical_or(tmp == SOC.STAIRS_DOWN, tmp == SOC.GROUND), axis=1)
                down_stairs_portal = np.array(down_stairs_portal)[tmp].mean(axis=0).astype(int)
                assert cat_map[down_stairs_portal[0], down_stairs_portal[1]] == SOC.GROUND
            else:
                down_stairs_portal = None
        else:
            down_stairs_portal = None

        return room_connections, room_door_grids, up_stairs_portal, down_stairs_portal

    def load_mesh_file(self):
        for floor_num in range(self.building.num_floors):
            floor = self.building.floors[floor_num]
            print(f"Processing floor {floor_num}...")
            # generate ray requests to extract room and object planform polygons
            room_ids = []
            room_seg_requests = []
            room_scene_categories = dict()  # for visualization-only
            room_obj_ids = []
            room_obj_seg_requests = []
            obj_classes = dict()
            for room_id, room in floor.rooms.items():
                assert room.floor_number == floor_num
                room_id = int(room_id)
                if room.scene_category == "staircase":
                    continue
                room_ids.append(room_id)
                room_seg_requests.append(self.extract_room_planform_polygon.remote(self, room_id))
                if self.debug:
                    print(f"Extracting room {room_id} {room.scene_category} on floor {floor_num}...")
                    ray.get(room_seg_requests[-1])
                room_scene_categories[room_id] = room.scene_category

                obj_ids = []
                obj_seg_requests = []
                for obj_id in room.objects.keys():
                    obj_id = int(obj_id)
                    obj_ids.append(obj_id)
                    obj_seg_requests.append(self.extract_object_planform_polygon.remote(self, obj_id))
                    if self.debug:
                        print(f"Extracting object {obj_id} {room.objects[obj_id].class_} on floor {floor_num}...")
                        ray.get(obj_seg_requests[-1])
                    obj_classes[obj_id] = room.objects[obj_id].class_
                if len(obj_ids) == 0:
                    continue
                room_obj_ids.append(obj_ids)
                room_obj_seg_requests.append(obj_seg_requests)

            # extract floor ground
            (
                floor_polygon,
                wall_polygons,
                ground_z,
                down_stairs_room_id,
                down_stairs_polygon,
                up_stairs_room_id,
                up_stairs_polygon,
            ) = self.extract_floor_ground(floor)
            floor.ground_z = ground_z
            floor.down_stairs_id = down_stairs_room_id
            if down_stairs_room_id >= 0:
                floor.down_stairs_uuid = self.building.floors[floor_num - 1].rooms[down_stairs_room_id].uuid
            floor.up_stairs_id = up_stairs_room_id
            if up_stairs_room_id >= 0:
                floor.up_stairs_uuid = floor.rooms[up_stairs_room_id].uuid

            # extract room segmentation
            print("Extracting room segmentation...")
            room_seg_results = ray.get(room_seg_requests)
            floor_map = floor.grid_map
            room_map = floor_map.get_room_map()
            for room_id, polygon_xy in zip(room_ids, room_seg_results):
                polygon_rc = floor_map.xy_to_rc(polygon_xy)  # (N, 2), (row, col)
                room_map = cv2.drawContours(room_map, [polygon_rc[:, ::-1]], -1, room_id, -1)
                floor.rooms[room_id].grid_map_min = polygon_rc.min(axis=0)
                floor.rooms[room_id].grid_map_max = polygon_rc.max(axis=0)
            if down_stairs_polygon is not None:
                down_stairs_polygon_rc = floor_map.xy_to_rc(get_polygon_xy(down_stairs_polygon))[:, ::-1]
                room_map = cv2.drawContours(room_map, [down_stairs_polygon_rc], -1, [down_stairs_room_id], -1)
                room_scene_categories[down_stairs_room_id] = "staircase-down"
            if up_stairs_polygon is not None:
                up_stairs_polygon_rc = floor_map.xy_to_rc(get_polygon_xy(up_stairs_polygon))[:, ::-1]
                room_map = cv2.drawContours(room_map, [up_stairs_polygon_rc], -1, [up_stairs_room_id], -1)
                room_scene_categories[up_stairs_room_id] = "staircase-up"
            floor_map.set_room_map(room_map)

            # extract object segmentation
            print("Extracting object segmentation...")
            cat_map = floor_map.get_cat_map()
            cat_map[room_map > 0] = SOC.WALL
            floor_polygon_rc = floor_map.xy_to_rc(floor_polygon)[:, ::-1]  # (N, 2), (col, row)
            cat_map = cv2.drawContours(cat_map, [floor_polygon_rc], -1, SOC.GROUND, -1)

            if len(wall_polygons) > 1 and shapely.intersection(wall_polygons[0], wall_polygons[1]).area > 0:
                outer_polygon = floor_map.xy_to_rc(get_polygon_xy(wall_polygons[0]))[:, ::-1]
                inner_polygon = wall_polygons[1].buffer(-self.wall_polygon_buffer.get(self.env_name, 0.0))
                inner_polygon = floor_map.xy_to_rc(get_polygon_xy(inner_polygon))[:, ::-1]
                cat_map = cv2.drawContours(cat_map, [outer_polygon], -1, SOC.WALL, -1)
                cat_map = cv2.drawContours(cat_map, [inner_polygon], -1, SOC.GROUND, -1)
            else:
                for wall_polygon in wall_polygons:
                    wall_polygon = get_polygon_xy(wall_polygon.buffer(self.wall_polygon_buffer.get(self.env_name, 0.0)))
                    cat_map = cv2.drawContours(cat_map, [floor_map.xy_to_rc(wall_polygon)[:, ::-1]], -1, SOC.WALL, -1)

            if up_stairs_polygon is not None:
                up_stairs_polygon_rc = up_stairs_polygon.buffer(self.up_stairs_buffer.get(self.env_name, 0.02))
                up_stairs_polygon_rc = floor_map.xy_to_rc(get_polygon_xy(up_stairs_polygon_rc))[:, ::-1]
                up_stairs_mask = np.zeros_like(cat_map)
                up_stairs_mask = cv2.drawContours(up_stairs_mask, [up_stairs_polygon_rc], -1, 1, -1)
                up_stairs_mask[cat_map == SOC.GROUND] = 0
                up_stairs_mask = up_stairs_mask == 1
                if not np.any(up_stairs_mask):
                    up_stairs_mask = np.zeros_like(cat_map)
                    up_stairs_mask = cv2.drawContours(up_stairs_mask, [up_stairs_polygon_rc], -1, 1, -1)
                    up_stairs_mask = up_stairs_mask == 1
                cat_map[up_stairs_mask] = SOC.STAIRS_UP
            if down_stairs_polygon is not None:
                down_stairs_polygon_rc = down_stairs_polygon.buffer(self.down_stairs_buffer.get(self.env_name, 0.02))
                down_stairs_polygon_rc = floor_map.xy_to_rc(get_polygon_xy(down_stairs_polygon_rc))[:, ::-1]
                down_stairs_mask = np.zeros_like(cat_map)
                down_stairs_mask = cv2.drawContours(down_stairs_mask, [down_stairs_polygon_rc], -1, 1, -1)
                down_stairs_mask[cat_map == SOC.GROUND] = 0
                down_stairs_mask = down_stairs_mask == 1
                if not np.any(down_stairs_mask):
                    down_stairs_mask = np.zeros_like(cat_map)
                    down_stairs_mask = cv2.drawContours(down_stairs_mask, [down_stairs_polygon_rc], -1, 1, -1)
                    down_stairs_mask = down_stairs_mask == 1
                cat_map[down_stairs_mask] = SOC.STAIRS_DOWN

            obj_id_to_result = dict()
            for obj_ids, obj_seg_requests in zip(room_obj_ids, room_obj_seg_requests):
                obj_seg_results = ray.get(obj_seg_requests)
                obj_seg_results = list(zip(obj_ids, obj_seg_results))
                obj_seg_results.sort(key=lambda x: x[1][1])
                for obj_id, (polygon_xy, z) in obj_seg_results:
                    polygon_rc = floor_map.xy_to_rc(polygon_xy)
                    cat_map = cv2.drawContours(cat_map, [polygon_rc[:, ::-1]], -1, obj_id, -1)
                    room_id = self.building.object_id_to_room_id[obj_id]
                    floor.rooms[room_id].objects[obj_id].grid_map_min = polygon_rc.min(axis=0)
                    floor.rooms[room_id].objects[obj_id].grid_map_max = polygon_rc.max(axis=0)
                    obj_id_to_result[obj_id] = (polygon_rc, z)

            # some objects are larger and higher, we need to draw smaller objects covered by larger objects
            all_obj_ids = list(obj_id_to_result.keys())
            map_obj_ids = np.unique(cat_map).tolist()
            disappeared_obj_ids = set(all_obj_ids) - set(map_obj_ids)
            while len(disappeared_obj_ids) > 0:
                for obj_id in disappeared_obj_ids:
                    polygon_rc = obj_id_to_result[obj_id][0]
                    cat_map = cv2.drawContours(cat_map, [polygon_rc[:, ::-1]], -1, obj_id, -1)
                map_obj_ids = np.unique(cat_map).tolist()
                disappeared_obj_ids = set(all_obj_ids) - set(map_obj_ids)

            floor_map.set_cat_map(cat_map)

            # get room connections
            print("Getting room connections...")
            (
                room_connections,
                room_door_grids,
                up_stairs_portal,
                down_stairs_portal,
            ) = self.find_room_connections(room_map, cat_map, up_stairs_room_id, down_stairs_room_id)
            for room_id, connected_room_ids in room_connections.items():
                if room_id == floor.down_stairs_id:
                    down_floor = self.building.floors[floor_num - 1]
                    down_stairs_room = down_floor.rooms[room_id]
                    for connected_room_id in connected_room_ids:
                        down_stairs_room.connected_room_ids.add(connected_room_id)
                        down_stairs_room.connected_room_uuids.add(floor.rooms[connected_room_id].uuid)
                else:
                    room = floor.rooms[room_id]
                    room.connected_room_ids = connected_room_ids
                    room.connected_room_uuids = set(
                        [
                            floor.rooms[connected_room_id].uuid
                            if connected_room_id != floor.down_stairs_id
                            else self.building.floors[floor_num - 1].rooms[connected_room_id].uuid
                            for connected_room_id in connected_room_ids
                        ]
                    )
                    room.door_grids.update(room_door_grids[room_id])

            floor.up_stairs_portal = up_stairs_portal
            floor.down_stairs_portal = down_stairs_portal

            floor.draw_room_map()
            floor.draw_cat_map(hold=self.debug)

        for floor_num in range(self.building.num_floors):
            floor = self.building.floors[floor_num]
            if floor_num > 0:  # compute down stairs cost
                down_floor = self.building.floors[floor_num - 1]
                portal1 = np.zeros(3)
                portal1[:2] = floor.grid_map.rc_to_xy(np.array(floor.down_stairs_portal))
                portal1[2] = floor.ground_z
                portal2 = np.zeros(3)
                portal2[:2] = down_floor.grid_map.rc_to_xy(np.array(down_floor.up_stairs_portal))
                portal2[2] = down_floor.ground_z
                floor.down_stairs_cost = np.linalg.norm(portal1 - portal2).item()
                down_floor.up_stairs_cost = floor.down_stairs_cost
            if floor_num < self.building.num_floors - 1:  # compute up stairs cost
                up_floor = self.building.floors[floor_num + 1]
                portal1 = np.zeros(3)
                portal1[:2] = floor.grid_map.rc_to_xy(np.array(floor.up_stairs_portal))
                portal1[2] = floor.ground_z
                portal2 = np.zeros(3)
                portal2[:2] = up_floor.grid_map.rc_to_xy(np.array(up_floor.down_stairs_portal))
                portal2[2] = up_floor.ground_z
                cost = np.linalg.norm(portal1 - portal2).item()
                if np.isinf(floor.up_stairs_cost):
                    floor.up_stairs_cost = cost
                else:
                    assert floor.up_stairs_cost == cost
                if np.isinf(up_floor.down_stairs_cost):
                    up_floor.down_stairs_cost = cost
                else:
                    assert up_floor.down_stairs_cost == cost


def draw_nx_graph(graph: nx.Graph, node_colors: list) -> np.ndarray:
    dpi = 250
    graph_fig = plt.figure(figsize=(1280 / dpi, 1440 / dpi), dpi=dpi)
    graph_ax = graph_fig.add_subplot(111)
    nx.draw(
        graph,
        ax=graph_ax,
        with_labels=True,
        font_size=4,
        node_size=400,
        pos=nx.nx_agraph.graphviz_layout(graph),
        node_color=node_colors,
    )
    plt.tight_layout()
    graph_fig.canvas.flush_events()
    graph_fig.canvas.draw()
    graph_img = np.frombuffer(graph_fig.canvas.renderer.buffer_rgba(), np.uint8)
    graph_img = graph_img.reshape(graph_fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(graph_fig)
    return graph_img[..., :3]


def generate_animation(building: Building, npz_file: str, mesh_file: str, output_path: str, fps: int = 30):
    print(f"Generating animation for building {building.name}...")
    print(f"Output path: {output_path}, FPS: {fps}")
    npz_data = np.load(npz_file, allow_pickle=True)["output"].item()
    room_inst_segmentation = npz_data["building"]["room_inst_segmentation"]
    object_inst_segmentation = npz_data["building"]["object_inst_segmentation"]
    mesh_data = trimesh.load(mesh_file)
    face_room_cat = room_inst_segmentation[:, 0].astype(int)
    obj_cat = object_inst_segmentation[:, 0].astype(int)
    vertices = mesh_data.vertices[:, [0, 2, 1]]
    vertices[:, 1] *= -1
    faces = mesh_data.faces
    faces_pos = np.mean(vertices[faces], axis=1)
    color_iter = iter(COLOR_MAP)

    cameras = {
        "Allensville": dict(
            position=(-6.45028, -12.0217, 26.2674),
            focal_point=(2.42836, 1.16739, 2.37249),
            viewup=(0.448430, 0.701543, 0.553849),
            roll=37.8254,
            distance=28.7010,
            clipping_range=(22.5667, 42.6550),
        ),
        "Benevolence": dict(
            position=(-7.91639, -14.0253, 24.5295),
            focal_point=(-0.226507, -2.11346, -0.424859),
            viewup=(0.468601, 0.732377, 0.494001),
            distance=28.7010,
            clipping_range=(22.4964, 36.9435),
        ),
        "Collierville": dict(
            position=(-10.5687, -11.6781, 24.9047),
            focal_point=(-0.434637, 2.90949, 2.36022),
            viewup=(0.420972, 0.663472, 0.618537),
            roll=39.6102,
            distance=28.7010,
            clipping_range=(25.1450, 40.7483),
        ),
    }

    camera = cameras[building.name]
    plotter = vedo.Plotter(axes=0, interactive=False, size=(1280, 1440))
    plotter_shown = False
    delay_cnt = 10
    vedo_meshes = []

    plt_fig = plt.figure(figsize=(2560 / 250, 1440 / 250), dpi=250)
    plt_ax = plt_fig.add_subplot(111)
    plt_ax.axis("off")
    plt_img = plt_ax.imshow(np.zeros((1440, 2560, 3), dtype=np.uint8))
    plt.tight_layout()
    plt_fig.canvas.flush_events()
    plt_fig.canvas.draw()
    plt.pause(0.01)

    graph = nx.Graph()
    node_colors = ["#1f78b4"]
    building_label = f"Building\n{building.name}"

    with imageio.get_writer(output_path, fps=fps) as writer:
        for floor_num in range(building.num_floors):
            floor = building.floors[floor_num]

            # graph img
            floor_label = f"Floor {floor_num}"
            graph.add_node(floor_label)
            graph.add_edge(building_label, floor_label)
            node_colors.append("#33a02c")
            graph_img = draw_nx_graph(graph, node_colors)

            # mesh img
            floor_mask = np.zeros_like(face_room_cat)
            for room in floor.rooms.values():
                floor_mask = np.logical_or(floor_mask, face_room_cat == int(room.id))
            floor_mask = np.logical_and(floor_mask, obj_cat == 0)
            floor_vertices, floor_faces = prune_unused_vertices(vertices, faces[floor_mask])
            floor_faces_pos = np.mean(floor_vertices[floor_faces], axis=1)
            z = floor_faces_pos[:, 2]
            shell_z_min = z.min()
            ground_z_max = shell_z_min + 0.3
            non_ceiling_mask = z <= ground_z_max + 1.3
            non_ceiling_vertices, non_ceiling_faces = prune_unused_vertices(
                floor_vertices,
                floor_faces[non_ceiling_mask],
            )
            non_ceiling_mesh: vedo.Mesh = vedo.Mesh([non_ceiling_vertices, non_ceiling_faces]).c(next(color_iter))
            vedo_meshes.append(non_ceiling_mesh)
            if plotter_shown:
                plotter += non_ceiling_mesh
                plotter.render()
            else:
                plotter.show(
                    non_ceiling_mesh,
                    non_ceiling_mesh.flagpost(f"Floor {floor_num}"),
                    interactive=False,
                    camera=camera,
                )
                plotter_shown = True
            mesh_img = plotter.screenshot(asarray=True)
            image = np.hstack([mesh_img, graph_img])
            for _ in range(delay_cnt):  # repeat for 3 times to make the image stays for a while
                writer.append_data(image)
            plt_img.set_data(image)
            plt_fig.canvas.flush_events()
            plt.pause(0.01)

            non_ceiling_mask = np.logical_and(floor_mask, faces_pos[:, 2] <= ground_z_max + 1.3)
            for room_id, room in floor.rooms.items():
                # graph img
                room_label = f"Room {room_id}\n{room.scene_category}"
                # nodes.append(room_label)
                # room_nodes.append(room_label)
                graph.add_node(room_label)
                graph.add_edge(floor_label, room_label)
                node_colors.append("#e31a1c")
                graph_img = draw_nx_graph(graph, node_colors)

                # mesh img
                room_mask = face_room_cat == int(room_id)
                room_mask = np.logical_and(room_mask, non_ceiling_mask)
                room_vertices, room_faces = prune_unused_vertices(vertices, faces[room_mask])
                color = next(color_iter)
                room_mesh: vedo.Mesh = vedo.Mesh([room_vertices, room_faces]).c(color)
                vedo_meshes.append(room_mesh)
                plotter += room_mesh
                plotter += room_mesh.flagpost(f"{room.uuid}: {room.scene_category}", bc=color)
                plotter.render()
                mesh_img = plotter.screenshot(asarray=True)
                image = np.hstack([mesh_img, graph_img])
                for _ in range(delay_cnt):  # repeat for 3 times to make the image stays for a while
                    writer.append_data(image)
                plt_img.set_data(image)
                plt_fig.canvas.flush_events()
                plt.pause(0.01)

                for obj_id, obj in room.objects.items():
                    # graph img
                    obj_label = f"Obj. {obj_id}\n{obj.class_}"
                    graph.add_node(obj_label)
                    graph.add_edge(room_label, obj_label)
                    node_colors.append("#ff7f00")
                    graph_img = draw_nx_graph(graph, node_colors)

                    obj_mask = obj_cat == int(obj_id)
                    obj_vertices, obj_faces = prune_unused_vertices(vertices, faces[obj_mask])
                    color = next(color_iter)
                    obj_mesh: vedo.Mesh = vedo.Mesh([obj_vertices, obj_faces]).c(color)
                    vedo_meshes.append(obj_mesh)
                    plotter += obj_mesh
                    plotter += obj_mesh.flagpost(f"{obj.uuid}: {obj.class_}", bc=color)
                    plotter.render()
                    mesh_img = plotter.screenshot(asarray=True)
                    image = np.hstack([mesh_img, graph_img])
                    for _ in range(delay_cnt):  # repeat for 3 times to make the image stays for a while
                        writer.append_data(image)
                    plt_img.set_data(image)
                    plt_fig.canvas.flush_events()
                    plt.pause(0.01)

            ceiling_mask = z > ground_z_max + 1.3
            ceiling_vertices, ceiling_faces = prune_unused_vertices(floor_vertices, floor_faces[ceiling_mask])
            ceiling_mesh = vedo.Mesh([ceiling_vertices, ceiling_faces]).c("gray")
            vedo_meshes.append(ceiling_mesh)
            plotter += ceiling_mesh
            plotter.render()
            mesh_img = plotter.screenshot(asarray=True)
            image = np.hstack([mesh_img, graph_img])
            for _ in range(delay_cnt):  # repeat for 3 times to make the image stays for a while
                writer.append_data(image)
            plt_img.set_data(image)
            plt_fig.canvas.flush_events()
            plt.pause(0.01)
    plt.close(plt_fig)


def load_building(
        npz_file: str,
        mesh_file: str,
        cell_size: float = 0.05,
        debug: bool = False,
) -> Building:
    loader = BuildingLoader(npz_file, mesh_file, cell_size, debug)
    return loader.building


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-file", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument(
        "--scene-graph-npz-file",
        type=str,
        required=True,
        help="Path to the .npz file of dense scene graph labeling.",
    )
    parser.add_argument("--output-dir", type=str, default=os.path.abspath(os.curdir), help="Directory to save results.")
    parser.add_argument("--ignore-pkl", action="store_true", help="Ignore pkl file in OUTPUT_DIR if specified.")
    parser.add_argument("--debug", action="store_true", help="Print and visualize intermediate results.")
    parser.add_argument("--save-images", action="store_true", help="Save images of floor layouts at the end.")
    parser.add_argument("--save-video", action="store_true", help="Save videos of graph generation at the end.")
    parser.add_argument("--fps", type=int, default=30, help="FPS of the generated animation. Default: 30.")
    args = parser.parse_args()

    mesh_file = args.mesh_file
    scene_graph_npz_file = args.scene_graph_npz_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    pkl_file = os.path.join(output_dir, "building.pkl")
    if os.path.isfile(pkl_file) and not args.ignore_pkl:
        with open(pkl_file, "rb") as file:
            building = pickle.load(file)
    else:
        building = load_building(scene_graph_npz_file, mesh_file, cell_size=0.01, debug=args.debug)
        with open(pkl_file, "wb") as file:
            pickle.dump(building, file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    building.save(output_dir)

    if args.save_images:
        for floor_num in range(building.num_floors):
            floor = building.floors[floor_num]
            floor.draw_room_map(os.path.join(output_dir, f"room_map_{floor.floor_num}.png"))
            floor.draw_cat_map(os.path.join(output_dir, f"cat_map_{floor.floor_num}.png"))

        building.visualize_as_graph(os.path.join(output_dir, "graph.png"), show=True)

    if args.save_video:
        generate_animation(
            building,
            scene_graph_npz_file,
            mesh_file,
            os.path.join(output_dir, "building_loader.mp4"),
            fps=args.fps,
        )


if __name__ == "__main__":
    print(f"running script {__file__}")
    main()
