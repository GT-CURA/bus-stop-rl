from dataclasses import dataclass

@dataclass
class Pic:
    """ Represents pictues, of which there can be multiple for a given POI. You 
    Probably don't need to interact with these. """
    heading: float
    lat: float
    lng: float
    pano_id = None
    date = None
    zoom_lvl = 0

    def get_coords(self):
        return f"{self.lat},{self.lng}"
    
    def get_key(self):
        return f"{self.pano_id}_{round(self.heading,1)}_{self.zoom_lvl}"
    
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
    side: str
    key: str
    date: str

@dataclass
class Hypothesis: 
    observations: list[Detection]
    triangulated_pos: tuple[float, float]
    score: float
    label: str
    last_seen: int
    best_bearing: float
    prev_bearing_err: float
    side: str

class Node:
    def __init__(self, lat, lng):
        # Location
        self.lat = lat
        self.lng = lng

        # Neighbor pano IDs
        self.neighbors = set()

        # Detections and scorecard
        self.best_conf = 0.0
        self.best_bearing = None
        self.scores = {"shelter": 0.0, "sign": 0.0, "trash can": 0.0, "seating": 0.0}

        # Detections (class in objects module) 
        self.detections = []
        self.det_count = {}

        # Visit counter
        self.visits = 0
    
    def add_det(self, det: Detection):
        # Add detection to node
        self.detections.append(det)
        
        if det.label in ["shelter", "sign"]:
            # Keep track of how many times this det has been found 
            if det.key in self.det_count:
                self.det_count[det.key] += 1
            else:
                self.det_count[det.key] = 1
            return self.det_count[det.key]
        else:
            return 0
