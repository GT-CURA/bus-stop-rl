from dataclasses import dataclass

@dataclass
class Pic:
    """ Used to pass info about a frame's position between methods/classes. """
    heading: float
    lat: float
    lng: float
    pano_id = None
    date = None
    zoom_lvl = 0

    def get_coords(self):
        return f"{self.lat},{self.lng}"

@dataclass
class Stop:
    og_lat: float
    og_lng: float
    place_name: str
    viewpoints: list
    false_negative: bool
    heading: float

    def get_coords(self):
        return str(self.og_lat) + "," + str(self.og_lng)

@dataclass
class Detection:
    bearing: float
    primary_conf: float
    box_sz: float
    cx_norm: float
    label: str
    timestamp: int
    pano_id: str
    lat: float
    lng: float
    local_x: float
    local_y: float

@dataclass
class Hypothesis: 
    observations: list[Detection]
    triangulated_pos: tuple[float, float]
    score: float
    label: str
    last_seen: int
    best_bearing: float
    prev_bearing_err: float