import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from llm_planning.scene_graph.color_map import COLOR_MAP
from llm_planning.scene_graph.special_object_categories import SOC


def draw_room_map(
    room_map: np.ndarray,
    room_categories: dict,
    room_uuids: dict,
    room_connections: list = None,
    room_door_grids: dict = None,
    title: str = "",
    output_path: str = None,
    show: bool = False,
    hold: bool = False,
):
    room_ids = np.unique(room_map).astype(int)
    room_ids.sort()
    n_colors = len(room_ids)
    img = np.zeros_like(room_map)
    labels = []
    room_uuids[-5] = "N/A"
    for i, room_id in enumerate(room_ids):
        img[room_map == room_id] = i
        labels.append(f'{room_id}/{room_uuids[room_id]}: {room_categories.get(room_id, "N/A")}')
    plt.figure(figsize=(10, 10))
    plt.imshow(img.T, cmap=ListedColormap(COLOR_MAP[:n_colors]))
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(n_colors)))
    cbar.ax.set_yticklabels(labels)
    if room_connections is not None:
        for loc1, loc2 in room_connections:
            plt.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], "k")
    if room_door_grids is not None:
        for room_id, door_grids in room_door_grids.items():
            for door_grid in door_grids.values():
                door_grid = np.array(door_grid).T
                plt.scatter(door_grid[0], door_grid[1], s=0.5, c="b")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    if show:
        if hold:
            plt.show()
        else:
            plt.pause(0.1)
    return img


def draw_cat_map(
    cat_map: np.ndarray,
    obj_categories: dict,
    obj_uuids: dict,
    up_stairs_portal: np.ndarray,
    down_stairs_portal: np.ndarray,
    room_door_grids: dict = None,
    title: str = "",
    output_path: str = None,
    show: bool = False,
    hold: bool = False,
):
    cat_ids = np.unique(cat_map).astype(int)
    cat_ids.sort()
    n_colors = len(cat_ids)
    img = np.zeros_like(cat_map)
    labels = []
    for i in range(-5, 1):
        obj_uuids[i] = "N/A"
    for i, cat_id in enumerate(cat_ids):
        img[cat_map == cat_id] = i
        if cat_id <= 0:  # special object category
            labels.append(f"{cat_id}/{obj_uuids[cat_id]}: {SOC.get(cat_id)}")
        else:
            labels.append(f"{cat_id}/{obj_uuids[cat_id]}: {obj_categories.get(cat_id, str(cat_id))}")
    plt.figure(figsize=(10, 10))
    plt.imshow(img.T, cmap=ListedColormap(COLOR_MAP[:n_colors]))
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(n_colors)))
    cbar.ax.set_yticklabels(labels)
    if up_stairs_portal is not None:
        plt.plot(up_stairs_portal[0], up_stairs_portal[1], "r*", markersize=10)
    if down_stairs_portal is not None:
        plt.plot(down_stairs_portal[0], down_stairs_portal[1], "r*", markersize=10)
    if room_door_grids is not None:
        for room_id, door_grids in room_door_grids.items():
            for door_grid in door_grids.values():
                door_grid = np.array(door_grid).T
                plt.scatter(door_grid[0], door_grid[1], s=0.5, c="b")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    if show:
        if hold:
            plt.show()
        else:
            plt.pause(0.1)
    return img
