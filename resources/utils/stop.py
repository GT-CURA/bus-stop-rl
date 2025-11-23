from dataclasses import dataclass

@dataclass
class Viewpoint:
    lon: float
    lat: float
    heading: float
    score: float
    boxes: list

@dataclass
class Stop:
    og_lat: float
    og_lng: float
    place_name: str
    viewpoints: list
    false_negative: bool
    heading: float

    def calc_cords(self):
        # Get two highest scoring viewpoints (or some other criteria)

        # Get the bounding boxes from viewpoints (either biggest primary evidence or highest scoring)

        # Estimate coordinate position of bus stop 
        pass

    def get_coords(self):
        return str(self.og_lat) + "," + str(self.og_lng)
