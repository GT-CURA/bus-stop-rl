from settings import S
import numpy as np
import math
from src.streetview.move import Move
from src.utils.objects import Stop, Pic
from src.streetview.sv_requests import Reqs
from src.streetview.graph_cache import GraphCache
from src.utils.tools import haversine

class StreetView:
    def __init__(self):
        self.reqs = Reqs()
        self.current_img = None
        self.current_stop: Stop
        self.current_pic: Pic
        self.start_state = None
        self.graph_cache = GraphCache()

    def goto_pt(self, stop: Stop = None):
        # Return if failed to load stop
        if not stop:
            return False
        self.current_stop = stop

        # Build Move class (pull OSMNX)
        self.move = Move(self.graph_cache, stop.og_lat, stop.og_lng)

        # Snap spawnpoint to road (don't spawn in parking lot)
        snapped = self.move.snap_pt(
            stop.og_lat,
            stop.og_lng,
        )

        # Use snapped pos if it succeeded, else inputted cords
        if snapped:
            lat, lng = snapped
        else:
            lat, lng = stop.og_lat, stop.og_lng

        # Build pic 
        pic = Pic(
            heading=None,
            lat=lat,
            lng=lng,
        )
        self.current_pic = pic

        # Pull pano ID if we don't have it 
        if self.current_pic.pano_id is None:
            self.reqs.pull_pano_info(self.current_pic)

        # Calculate heading
        if self.current_pic.heading is None:
            self._estimate_heading(self.current_pic, stop)

        # Pull image
        self.current_img = self.reqs.pull_image(self.current_pic)

        # Calculate initial road vectors
        self.move.calc_rd_vectors(self.current_pic)

        return True


    def get_img(self):
        return self.current_img

    def do_action(self, action, pull_img = True):
        """ Immitate movement in streetview. """
        # Rotate counterclockwise
        if action == 'Counterclockwise':
            self.current_pic.heading -= S.rotate_amt
            self.current_pic.heading = self.current_pic.heading % 360
            self.current_pic.zoom_lvl = 0
            
        # Rotate clockwise
        elif action == 'Clockwise':
            self.current_pic.heading += S.rotate_amt
            self.current_pic.heading = self.current_pic.heading % 360
            self.current_pic.zoom_lvl = 0

        # Handle movement with Move class
        elif action == 'Forwards':
            self.current_pic = self.move.move(self.current_pic, backwards=False)
        
        elif action == 'Backwards':
            self.current_pic = self.move.move(self.current_pic, backwards=True)

        # Zoom in
        elif action == "Zoom":
            self._zoom()

        # Pull image if requested
        if pull_img:
            
            # Pull new pic
            zoom = True if self.current_pic.zoom_lvl > 0 else False
            self.current_img = self.reqs.pull_image(self.current_pic, zoom)

        # Otherwise, just do metadata call
        else:
            if not self.current_pic.pano_id:
                self.reqs.pull_pano_info(self.current_pic)
    
    def check_cache(self):
        dist = haversine(self.current_pic.lat, self.current_pic.lng, )

    def goto_start(self):
        """ Go back to the initial position. """
        # Create copy of the original pos
        self.current_pic = Pic(
            heading = self.start_state["heading"],
            lat=self.start_state["lat"],
            lng=self.start_state["lng"]
        )
        self.current_pic.pano_id = self.start_state["pano_id"]
        self.current_pic.zoom_lvl = self.start_state["zoom_lvl"]
        self.current_pic.date = self.start_state["date"]

        # Pull image
        self.current_img = self.reqs.pull_image(self.current_pic)

    def set_start(self):
        """ Log the current Pic's attributes, setting it as the starting point """
        self.start_state = {
            'heading': self.current_pic.heading,
            'lat': self.current_pic.lat,
            'lng': self.current_pic.lng,
            'pano_id': self.current_pic.pano_id,
            'zoom_lvl': self.current_pic.zoom_lvl,
            'date': self.current_pic.date
        }

    def _zoom(self):
        # See if we're at max zoom level
        if self.current_pic.zoom_lvl == 2:
            return
        
        # Otherwise, increase zoom level 
        self.current_pic.zoom_lvl += 1
        return
        
    def _estimate_heading(self, pic, stop: Stop):
        """
        Use pano's coords to determine the necessary camera heading.
        """
        # Convert latitude to radians, get distance between pic & POI lons in radians.  
        diff_lng = math.radians(stop.og_lng - pic.lng)
        old_lat = math.radians(pic.lat)
        new_lat = math.radians(stop.og_lat)

        # Determine degree bearing
        x = math.sin(diff_lng) * math.cos(new_lat)
        y = math.cos(old_lat) * math.sin(new_lat) - math.sin(old_lat) * math.cos(new_lat) * math.cos(diff_lng)
        heading = math.atan2(x, y)
        
        # Convert from radians to degrees, normalize
        heading = math.degrees(heading)
        heading = (heading + 360) % 360
        pic.heading = heading

    def calc_street_side(self, det):

        # Get road vectors
        perp = self.move.get_road_vec()

        # NOTE: Fix this 
        if perp is None: 
            det.side = "right"
            return 

        # Camera's location 
        cam_x, cam_y = det.local_x, det.local_y

        # Convert bearing to world-space direction vector where 0 degress is north
        theta = np.radians(det.bearing)
        dir_vec = np.array([np.sin(theta), np.cos(theta)])

        # Estimate object XY just to differentiate from objects on other side of the road
        obj_x = cam_x + 20 * dir_vec[0]
        obj_y = cam_y + 20 * dir_vec[1]

        # Vector to object
        to_obj = np.array([obj_x - cam_x, obj_y - cam_y])
        to_obj /= (np.linalg.norm(to_obj) + 1e-9)

        # Classify side
        side_value = np.dot(to_obj, perp)
        det.side = "left" if side_value > 0 else "right"