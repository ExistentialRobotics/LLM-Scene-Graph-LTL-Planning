class SOC:  # special object categories
    GROUND = 0
    STAIRS_UP = -1
    STAIRS_DOWN = -2
    WALL = -3
    CEILING = -4
    NA = -5

    @staticmethod
    def get(cat_id: int) -> str:
        return [
            "GROUND",
            "STAIRS_UP",
            "STAIRS_DOWN",
            "WALL",
            "CEILING",
            "N/A",
        ][abs(cat_id)]
