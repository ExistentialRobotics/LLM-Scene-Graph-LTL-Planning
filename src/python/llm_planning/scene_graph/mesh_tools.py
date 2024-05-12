import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
import trimesh
from matplotlib import cm

from llm_planning.scene_graph.grid_map import GridMap

CHAR_STR = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_DICT = {key: value for (key, value) in zip(range(len(CHAR_STR)), CHAR_STR)}


def load_mesh(mesh_path):
    return trimesh.load(mesh_path)


def extract_face_segmentation(npz_path, vertices, faces, building, floor):
    data = np.load(npz_path, allow_pickle=True)["output"].item()
    face_pos = np.mean(vertices[faces, :], axis=1)
    face_pos[:, [1, 2]] = face_pos[:, [2, 1]]
    face_room_cat = data["building"]["room_inst_segmentation"][:, 0]
    face_obj_cat = data["building"]["object_inst_segmentation"][:, 0]

    floor_mask = np.zeros_like(face_room_cat)
    for room in building.rooms:
        if floor == building.rooms[room].floor_number:
            floor_mask[face_room_cat == room] = 1

    floor_mask = floor_mask.astype(bool)

    face_pos = face_pos[floor_mask, :]
    face_room_cat = face_room_cat[floor_mask]
    face_obj_cat = face_obj_cat[floor_mask]

    shell_face_inds = np.where(face_obj_cat == 0)[0]
    shell_faces = face_pos[shell_face_inds, :]
    shell_z_min = np.min(shell_faces[:, 2])
    shell_z_max = np.max(shell_faces[:, 2])
    bin_size = (shell_z_max - shell_z_min) / 10
    ground_shell_inds = np.logical_and(shell_z_min <= shell_faces[:, 2], shell_faces[:, 2] <= (shell_z_min + bin_size))
    wall_shell_inds = np.logical_and(
        (shell_z_min + bin_size) <= shell_faces[:, 2], shell_faces[:, 2] <= (shell_z_min + 3 * bin_size)
    )
    ceiling_shell_inds = np.logical_not(np.logical_or(ground_shell_inds, wall_shell_inds))
    face_obj_cat[shell_face_inds[ground_shell_inds]] = 0
    face_obj_cat[shell_face_inds[ceiling_shell_inds]] = -2
    face_obj_cat[shell_face_inds[wall_shell_inds]] = -1

    return face_pos, face_room_cat, face_obj_cat


def get_grid_map(face_pos, face_room_cat, face_obj_cat, cell_size=0.5):
    face_mask = np.where(face_obj_cat != -2)[0]
    face_pos_m = face_pos[face_mask, :]
    face_room_cat_m = face_room_cat[face_mask]
    face_obj_cat_m = face_obj_cat[face_mask]
    x_min = np.min(face_pos_m[:, 0])
    x_max = np.max(face_pos_m[:, 0])
    y_min = np.min(face_pos_m[:, 1])
    y_max = np.max(face_pos_m[:, 1])

    size_c = np.round((x_max - x_min) / cell_size).astype(int)
    size_r = np.round((y_max - y_min) / cell_size).astype(int)

    grid_map = GridMap(cell_size=cell_size, origin=np.array([x_min, y_max]), map_size=(size_r, size_c))

    face_rc = grid_map.xy_to_rc(face_pos_m[:, :2])

    face_r = face_rc[:, 0]
    face_c = face_rc[:, 1]

    for r in range(size_r):
        for c in range(size_c):
            rc_inds = np.logical_and(face_r == r, face_c == c)
            if rc_inds.any():
                grid_map.set_cat_element(r, c, sp.mode(face_obj_cat_m[rc_inds], keepdims=False)[0])
                grid_map.set_room_element(r, c, sp.mode(face_room_cat_m[rc_inds], keepdims=False)[0])

    return grid_map


def cleanup_map(grid_map, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    cat_map = grid_map.get_cat_map()

    free_mask = cat_map == 0
    free_mask = cv2.dilate(free_mask.astype(np.uint8), kernel=kernel, iterations=2)
    free_mask = cv2.erode(free_mask.astype(np.uint8), kernel=kernel, iterations=2).astype(bool)

    wall_mask = cat_map == -1
    wall_mask = cv2.dilate(wall_mask.astype(np.uint8), kernel=kernel, iterations=2)
    wall_mask = cv2.erode(wall_mask.astype(np.uint8), kernel=kernel, iterations=2).astype(bool)

    object_mask = cat_map > 0

    map_size = grid_map.get_map_size()

    cleaned_cat_map = np.ones(map_size) * (-2)

    cleaned_cat_map[free_mask] = 0
    cleaned_cat_map[wall_mask] = -1
    cleaned_cat_map[object_mask] = cat_map[object_mask]

    grid_map.set_cat_map(cleaned_cat_map)

    return grid_map


def find_room_connections(building, floor, kernel_size=3, min_intersection=3):
    grid_map = building.floor_maps[floor]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    room_boundary_masks = dict()

    room_map = grid_map.get_room_map()
    cat_map = grid_map.get_cat_map()
    for room in building.rooms:
        if building.rooms[room].floor_number == floor:
            room_mask = room_map == int(room)
            # plt.figure()
            # plt.imshow(room_mask)
            # plt.show()
            room_mask = cv2.dilate(room_mask.astype(np.uint8), kernel=kernel, iterations=2)
            room_mask = cv2.erode(room_mask.astype(np.uint8), kernel=kernel, iterations=2).astype(bool)

            # plt.figure()
            # plt.imshow(room_mask)
            # plt.show()

            room_boundary_mask = np.zeros_like(room_mask)
            room_boundary = cv2.findContours(room_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
            room_boundary_mask[room_boundary[:, 0, 1], room_boundary[:, 0, 0]] = 1
            # plt.figure()
            # plt.imshow(room_boundary_mask)
            # plt.show()
            wall_mask = cat_map == -1
            # plt.figure()
            # plt.imshow(wall_mask)
            # plt.show()
            wall_mask = cv2.filter2D(wall_mask.astype(np.uint8), -1, kernel) > 0
            # plt.figure()
            # plt.imshow(wall_mask)
            # plt.show()
            # plt.figure()
            # plt.imshow(room_boundary_mask)
            # plt.show()
            room_boundary_mask[wall_mask] = 0
            # plt.figure()
            # plt.imshow(room_boundary_mask)
            # plt.show()
            room_boundary_mask = cv2.filter2D(room_boundary_mask.astype(np.uint8), -1, kernel) > 0
            # plt.figure()
            # plt.imshow(room_boundary_mask)
            # plt.show()

            room_boundary_masks[room] = room_boundary_mask

    for room_i in building.rooms:
        if building.rooms[room_i].floor_number != floor:
            continue

        for room_j in building.rooms:
            if building.rooms[room_j].floor_number != floor:
                continue

            if room_i == room_j:
                continue

            boundary_i = room_boundary_masks[room_i]
            boundary_j = room_boundary_masks[room_j]

            if np.sum(boundary_i * boundary_j) > min_intersection:
                building.rooms[room_i].connections.add(room_j)

    return building


def draw_room_connections(building, resize_coef=10):
    for floor_num in range(building.num_floors):
        floor_char = CHAR_DICT[floor_num]
        grid_map = building.floor_maps[floor_char]

        room_map = grid_map.get_room_map()

        sm = cm.ScalarMappable(cmap=plt.get_cmap("seismic"))
        canvas = sm.to_rgba(room_map, bytes=True)
        map_size = grid_map.get_map_size()
        canvas = cv2.resize(
            canvas, (map_size[1] * resize_coef, map_size[0] * resize_coef), interpolation=cv2.INTER_NEAREST
        )

        for room in building.rooms:
            if building.rooms[room].floor_number != floor_char:
                continue

            for connection in building.rooms[room].connections:
                if building.rooms[connection].floor_number != floor_char:
                    continue

                # room_loc = grid_map.xy_to_rc(building.rooms[room].location[:2])
                # connection_loc = grid_map.xy_to_rc(building.rooms[connection].location[:2])

                room_loc = np.mean(np.argwhere(room_map == room) * resize_coef, axis=0).astype(int)
                connection_loc = np.mean(np.argwhere(room_map == connection) * resize_coef, axis=0).astype(int)

                canvas = cv2.line(
                    canvas,
                    (room_loc[1], room_loc[0]),
                    (connection_loc[1], connection_loc[0]),
                    (0, 255, 0),
                    1 * resize_coef,
                )

        plt.figure()
        plt.imshow(canvas)
        plt.title("Room Segmentation for Floor {floor}".format(floor=floor_char))
        plt.show()


def show_cat_map(building, resize_coef=10):
    for floor_num in range(building.num_floors):
        floor_char = CHAR_DICT[floor_num]
        grid_map = building.floor_maps[floor_char]

        sm = cm.ScalarMappable(cmap=plt.get_cmap("prism"))
        canvas = sm.to_rgba(grid_map.get_cat_map(), bytes=True)
        # canvas = grid_map.get_cat_map()
        map_size = grid_map.get_map_size()
        canvas = cv2.resize(
            canvas, (map_size[1] * resize_coef, map_size[0] * resize_coef), interpolation=cv2.INTER_NEAREST
        )

        plt.figure()
        plt.imshow(canvas)
        plt.title("Object Segmentation for Floor {floor}".format(floor=floor_char))
        plt.show()
