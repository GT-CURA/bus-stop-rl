from dataclasses import dataclass
import requests 
from resources.stop import Stop
from playwright.sync_api import sync_playwright
import json
from settings import S
import cv2
import numpy as np
import math
from pathlib import Path
from requests.exceptions import ReadTimeout
import time

class StreetView:
    def __init__(self):
        self.key: str
        self.reqs: Requests
        self.current_img = None
        self.start_stop: Stop
        self.current_stop: Stop
        self.current_pic: Pic

    def launch(self, key_path = "key.txt"):
        """ Read key and build requests class (handles interfacing with API)"""
        # Read key, start requests
        key = open(key_path, "r").read()
        self.reqs = Requests(key, [640,640])

        # Launch playwright browser
        browser = sync_playwright().start().chromium.launch(headless=True)
        self.page = browser.new_page()

        # Set key
        html_content = open("resources\link_fetcher.html").read()
        html_with_key = html_content.replace(key, "")
        self.page.set_content(html_with_key)

    def goto_pt(self, stop: Stop):
        """ Used by loader class to pull initial image of point. """
        # If this is the initial use, define starting stop
        self.current_stop = stop

        # Build pic, send img
        self.current_pic = Pic(
            heading=None,
            lat=stop.og_lat,
            lng=stop.og_lng,
        )

        # Pull metadata request to find pano location
        self.reqs.pull_pano_info(self.current_pic)
        self._estimate_heading(self.current_pic, stop)

        # Pull image
        self.current_img = self.reqs.pull_image(self.current_pic)
        return True

    def get_img(self):
        """ Load bytes from streetview into CV2 image. """
        nparr = np.frombuffer(self.current_img, np.uint8)
        try:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error decoding image: {e}")
        cv2.imwrite("static/frame.jpg", img)
        return img

    def do_action(self, action):
        """ Immitate movement in streetview. """
        # Rotate counterclockwise
        if action == 'a':
            self.current_pic.heading -= S.rotate_amt
        
        # Rotate clockwise
        elif action == 'd':
            self.current_pic.heading += S.rotate_amt

        # Move forwards
        elif action == 'w':
            self._move('w')

        # Move backwards    
        elif action == 's':
            self._move('s')
        
        # Zoom in
        elif action == "=":
            self._zoom()

        # Pull new pic
        if self.current_pic.zoom_lvl > 0:
            self.current_img = self.reqs.old_pull_img(self.current_pic)
        else:
            self.current_img = self.reqs.pull_image(self.current_pic)
    
    def goto_start(self):
        """ Go back to the initial position. """
        self.current_pic = Pic(
            heading = self.start_heading,
            lat = self.start_stop.og_lat,
            lng = self.start_stop.og_lng
        )
        self.goto_pt(self.current_stop)

    def set_start(self):
        """ Basically tells class to reset. """
        self.start_stop = self.current_stop
        self.start_heading = self.current_pic.heading

    def _move(self, direction = 'w', dist = 4, tries = 0):
        # Reset zoom level 
        self.current_pic.zoom_lvl = 0

        # Reverse heading if necessary
        heading = self.current_pic.heading
        if direction != 'w':
            heading = self.current_pic.heading - 180

        # If tries have been surpassed, increase heading
        if tries > 2:
            if direction == 'w':
                heading += 70 
            else:
                heading -= 70

        # Calculate new coordinates
        earth_radius = 6378137
        heading_rad = math.radians(heading)
        new_lat = self.current_pic.lat + (dist / earth_radius) * math.cos(heading_rad) * (180 / math.pi)
        new_lng = self.current_pic.lng + (dist / earth_radius) * math.sin(heading_rad) * (180 / math.pi) / math.cos(math.radians(self.current_pic.lat))

        # Increment if pano ID equals current pano ID
        pic = Pic(self.current_pic.heading, new_lat, new_lng)

        # Attempt to pull pano info, return if nothing found 
        response = self.reqs.pull_pano_info(pic)
        if response == False: 
            return 
        
        # Ensure that we're pulling a new pano
        if pic.pano_id == self.current_pic.pano_id:
            self._move(direction=direction, dist=dist+4, tries=tries+1)
        else:
            self.current_pic=pic

    def _move_prev(self, direction='w'):
        pano_id_result = {}
        
        # Write to API counter
        path = Path(f"{S.log_dir}/api_calls.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # If file exists, read and increment
        if path.exists():
            with open(path, "r+") as f:
                try:
                    count = int(f.read())
                except ValueError:
                    count = 0
                count += 1
                f.seek(0)
                f.write(str(count))
                f.truncate()
        else:
            # Create file and initialize to 1
            with open(path, "w") as f:
                f.write("1")
                
        def handle_console(msg):
            try:
                text = msg.text
                # Strip quotes
                if text.startswith('"') and text.endswith('"'):
                    pano_id_result["pano_id"] = json.loads(text)
            except Exception:
                pass

        self.page.on("console", handle_console)

        if self.current_pic.pano_id:
            self.page.evaluate(f"""
                window.findNextPano(null, null, {self.current_pic.heading}, "{direction}", "{self.current_pic.pano_id}");
            """)
        else:
            self.page.evaluate(f"""
                window.findNextPano(
                {self.current_pic.lat}, {self.current_pic.lng}, {self.current_pic.heading}, "{direction}");
            """)

        # Build new pic with pano_id
        self.current_pic = Pic(
            heading=self.current_pic.heading,
            lat=None,
            lng=None,
        )
        self.current_pic.pano_id = pano_id_result.get("pano_id")

        # Get coordinates through metadata call
        self.reqs.pull_pano_info(self.current_pic)

    def _zoom(self):
        # See if we're at max zoom level
        if self.current_pic.zoom_lvl == 2:
            return
        
        # Otherwise, increase zoom level 
        self.current_pic.zoom_lvl += 1
        

    def _estimate_heading(self, pic, stop: Stop):
        """
        Use pano's coords to determine the necessary camera heading.
        """
        # Convert latitude to radians, get distance between pic & POI lons in radians.  
        diff_lon = math.radians(stop.og_lng - pic.lng)
        old_lat = math.radians(pic.lat)
        new_lat = math.radians(stop.og_lat)

        # Determine degree bearing
        x = math.sin(diff_lon) * math.cos(new_lat)
        y = math.cos(old_lat) * math.sin(new_lat) - math.sin(old_lat) * math.cos(new_lat) * math.cos(diff_lon)
        heading = math.atan2(x, y)
        
        # Convert from radians to degrees, normalize
        heading = math.degrees(heading)
        heading = (heading + 360) % 360
        pic.heading = heading
        
@dataclass
class Error:
    # I have OCD 
    context: str
    msg: str

    def __repr__(self):
        return f"{self.msg} while {self.context}."
    
    def alert(self, debug:bool):
        if debug: 
            print(f"[ERROR] {self.context} while {self.context}")

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

class Requests:
    def __init__(self, key: str, pic_dims, debug = False):
        self.key = key
        self.debug = debug
        if pic_dims:
            self.pic_len = pic_dims[0]
            self.pic_height = pic_dims[1]
    
    def old_pull_img(self, pic: Pic):
        path = Path(f"{S.log_dir}/api_calls.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # If file exists, read and increment
        if path.exists():
            with open(path, "r+") as f:
                try:
                    count = int(f.read())
                except ValueError:
                    count = 0
                count += 1
                f.seek(0)
                f.write(str(count))
                f.truncate()
        else:
            # Create file and initialize to 1
            with open(path, "w") as f:
                f.write("1")

        # Add zoom level (FOV)
        zoom_to_fov = {0: 90, 1: 60, 2: 30}
        fov = zoom_to_fov[pic.zoom_lvl]

        # Parameters for API request
        pic_params = {
            'key': self.key,
            'return_error_code': True,
            'outdoor': True,
            'size': f"{self.pic_len}x{self.pic_height}",
            'fov':fov}

        # Add either pano ID or location
        if pic.pano_id:
            pic_params['pano'] = pic.pano_id
        else:
            pic_params['location'] = pic.get_coords()

        # Add heading if there is any
        if pic.heading:
            pic_params['heading'] = pic.heading

        # Pull response 
        response = self._pull_response(
            params = pic_params,
            context = "Pulling image",
            coords = pic.get_coords(),
            base = 'https://maps.googleapis.com/maps/api/streetview?')
        
        # Close response, return content 
        content = response.content
        response.close()
        return content

    def pull_pano_info(self, pic: Pic):
        """
        Extract coordiantes from a pano's metadata, used to determine heading
        """
        # Params for request
        params = {
            'key': self.key,
            'return_error_code': True,
        }

        if pic.pano_id:
            params['pano'] = pic.pano_id
        else:
            params['location'] = pic.get_coords()
            
        # Send a request
        response = self._pull_response(
            params=params,
            coords=pic.get_coords(),
            context="Pulling metadata",
            base='https://maps.googleapis.com/maps/api/streetview/metadata?')
        
        # Handle finding no results
        if b'ZERO_RESULTS' in response.content:
            return False
        
        # Fetch the coordinates from the json response and store them in the POI
        pano_location = response.json().get("location")
        pic.lng = pano_location["lng"]
        pic.lat = pano_location["lat"]
        pic.pano_id = response.json().get("pano_id")
        pic.date = response.json().get("date")
        response.close()
        return True

    def _pull_response(self, params, context, base, coords):
        # Print a sumamry of the request if debugging 
        if self.debug: print(f"[REQUEST] {context} for {coords}")
        
        # Retry several times
        for i in range(10):
            try:
                response = requests.get(base, params=params, timeout=10)
                response.raise_for_status()
                return response
            except ReadTimeout:
                print(f"[Retry {i+1}] Waiting 5 seconds")
                time.sleep(5)

        # Check for empty response 
        if not response.content:
            return Error(context, "empty response")
        
        # Return error if the request was not successful
        else:
            response.close()
            return Error(context, f"({response.status_code}): {response.text}")