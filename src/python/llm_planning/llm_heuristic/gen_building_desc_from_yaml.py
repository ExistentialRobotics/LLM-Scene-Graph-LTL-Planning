import yaml
import os


def load_building_yaml(yaml_path: str):
    locations = {}
    room_names = {}
    with open(yaml_path, "r") as file:
        building_dict = yaml.load(file, Loader=yaml.FullLoader)
        desc = "You are a robot in building {}.\n".format(building_dict["name"])
        desc += "Building {} has {} floors and {} rooms.\n".format(
            building_dict["name"], building_dict["num_floors"], building_dict["num_rooms"]
        )
        for floor_key in building_dict["floors"].keys():
            floor = building_dict["floors"][floor_key]
            desc += "\tFloor {} has {} rooms: ".format(floor["id"], floor["num_rooms"])
            desc += ", ".join([str(room["uuid"]) for room_key, room in floor["rooms"].items()]) + ".\n"
            for room_key, room in floor["rooms"].items():
                desc += "\t\tRoom {} is a {} with ID {}.\n".format(room["uuid"], room["name"], room["uuid"])
                locations[room["uuid"]] = room["location"]
                room_names[room["uuid"]] = room["name"]
                desc += "\t\tRoom {} is connected to rooms: ".format(room["uuid"])
                desc += ", ".join([str(x) for x in room["connected_room_uuids"]]) + ".\n"
                if len(room["objects"]) > 0:
                    desc += "\t\tRoom {} has {} objects: ".format(room["uuid"], len(room["objects"]))
                    desc += (
                        ", ".join(
                            [
                                "a {} with ID {}".format(obj["name"], obj["uuid"], obj["parent_uuid"])
                                for obj in room["objects"].values()
                            ]
                        )
                        + ".\n"
                    )
                else:
                    desc += "\t\tRoom {} has 0 objects.\n".format(room["uuid"])
                for object_key, obj in room["objects"].items():
                    locations[obj["uuid"]] = obj["location"]

    return desc, room_names, locations


def test():
    load_path = "../../../assets/Benevolence/"
    # load_build = yaml.load('./data/Benevolence/building.yaml', Loader=yaml.FullLoader)
    yaml_file = os.path.join(load_path, f"building.yaml")
    desc, room_names, locations = load_building_yaml(yaml_file)
    print(desc)
    print("Loaded yaml file!")


if __name__ == "__main__":
    test()
