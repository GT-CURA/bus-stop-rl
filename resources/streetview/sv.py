from dataclasses import dataclass
import requests 
from resources.utils.objects import Stop
from requests.exceptions import RequestException
import json
from settings import S
import numpy as np
import math
from pathlib import Path
from requests.exceptions import RequestException, ReadTimeout, HTTPError, ConnectionError
import time
import cv2
import re
from random import uniform

class StreetView:
    def __init__(self):
        self.reqs: Requests
        self.current_img = None
        self.start_stop: Stop
        self.current_stop: Stop
        self.current_pic: Pic

    def launch(self, key_path = "key.txt"):
        """ Read key and build requests class (handles interfacing with API)"""
        # Read key, start requests
        self.reqs = Requests()

    def goto_pt(self, stop: Stop = None):
        """ Used by loader class to pull initial image of point. """
        # Go to the inputted stop (first use for episode)
        if stop:
            # If this is the initial use, define starting stop
            self.current_stop = stop

            # Build pic using stop's info, set as current pic
            self.current_pic = Pic(
                heading=None,
                lat=stop.og_lat,
                lng=stop.og_lng,
            )

        # Pull metadata request to find pano location
        if self.current_pic.pano_id == None:
            self.reqs.pull_pano_info(self.current_pic)
        if self.current_pic.heading == None:
            self._estimate_heading(self.current_pic, stop)

        # Pull image
        self.current_img = self.reqs.pull_image(self.current_pic)
        return True

    def get_img(self):
        """ Load bytes from streetview into CV2 image. """
        # Decode bytes into image, return it
        nparr = np.frombuffer(self.current_img, np.uint8)
        try:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error decoding image: {e}")
        return img

    def do_action(self, action, pull_img = True):
        """ Immitate movement in streetview. """
        # Rotate counterclockwise
        if action == 'Counterclockwise':
            self.current_pic.heading -= S.rotate_amt
            self.current_pic.heading = self.current_pic.heading % 360
            
        # Rotate clockwise
        elif action == 'Clockwise':
            self.current_pic.heading += S.rotate_amt
            self.current_pic.heading = self.current_pic.heading % 360

        # Move forwards
        elif action == 'Forwards':
            self._move('Forwards')

        # Move backwards    
        elif action == 'Backwards':
            self._move('Backwards')
        
        # Zoom in
        elif action == "Zoom":
            self._zoom()

        # Pull image if requested
        if pull_img:
            # Pull new pic
            if self.current_pic.zoom_lvl > 0:
                self.current_img = self.reqs.old_pull_img(self.current_pic)
            else:
                self.current_img = self.reqs.pull_image(self.current_pic)
        
        # Otherwise, just do metadata call
        else:
            if not self.current_pic.pano_id:
                self.reqs.pull_pano_info(self.current_pic)
    
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

        # Go to original pos
        self.goto_pt()

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

    def _move(self, direction = 'Forwards', heading = None):
        def _calc_coords(heading):
            # Calculate new coordinates
            earth_radius = 6378137
            heading_rad = math.radians(heading)
            new_lat = self.current_pic.lat + (S.dist / earth_radius) * math.cos(heading_rad) * (180 / math.pi)
            new_lng = self.current_pic.lng + (S.dist / earth_radius) * math.sin(heading_rad) * (180 / math.pi) / math.cos(math.radians(self.current_pic.lat))
            return Pic(self.current_pic.heading, new_lat, new_lng)
        
        # Reset zoom level 
        self.current_pic.zoom_lvl = 0

        # Reverse heading if necessary
        heading = self.current_pic.heading
        if direction != 'Forwards':
            heading = self.current_pic.heading - 180

        # Increment if pano ID equals current pano ID
        pic = _calc_coords(heading)

        # See if a new pano (or any) was found 
        response = self.reqs.pull_pano_info(pic)
        if response == False or pic.pano_id == self.current_pic.pano_id: 
    
            # If pano wasn't found, try to get street dir 
            street_dir = self.reqs.get_street_dir(self.current_pic)
            if street_dir:
                target_dir = self.current_pic.heading

                # Flip target dir if going backwards
                if direction == 'Backwards':
                    target_dir -= 180 
                
                # Normalize 
                target_dir = target_dir % 360
                opp_street_dir = (street_dir - 180) % 360
                street_dir =  street_dir % 360
                
                def angular_difference(a, b):
                    """Returns the smallest difference between two angles in degrees."""
                    return min(abs(a - b), 360 - abs(a - b))
                
                # Pick either street dir or opposite, depending on which is closer
                if angular_difference(target_dir, street_dir) <= angular_difference(target_dir, opp_street_dir):
                    new_heading = street_dir
                else:
                    new_heading = opp_street_dir
                pic = _calc_coords(new_heading)

            # If no sstreet dir, just add 70 degrees
            else:
                if direction == 'Forwards':
                    pic = _calc_coords(self.current_pic.heading + 70)
                else:
                    pic = _calc_coords(self.current_pic.heading - 70)

        # Update current pic
        self.current_pic = pic

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
    def __init__(self):
        self.max_uses_per_key = 9500
        self.keys = []
        self.current_key_index = 0
        self.counter_path = Path(f"{S.log_dir}/api_calls.json")
        
        # Create image cache 
        self.cache_dir = Path("cached_imgs")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load keys
        with open(S.key_path, "r") as f:
            self.keys = [k.strip() for k in f.read().split(",") if k.strip()]

        if not self.keys:
            raise ValueError("No API keys provided.")
        
        # Load call counts
        self._load_usage_counts()
        self._rotate_key()

    def pull_image(self, pic: Pic):
        # Check cache
        if S.img_caching:
            cached = self._pull_from_cache(pic)
            if cached:
                if S.request_msgs: print(f"[Cache] Hit for {pic.pano_id}")
            return cached
        
        # Get pano ID if we don't have it
        if not pic.pano_id:
            self.pull_pano_info(pic)

        url = f"https://streetviewpixels-pa.googleapis.com/v1/thumbnail?cb_client=maps_sv.tactile&w=640&h=640&panoid={pic.pano_id}&yaw={pic.heading}&pitch=0.00"
        response = self._request(url, context="Pulling Thumbnail")
        # If API returns error image or no content
        if response.status_code == 400:
            print("[Requests] Got 400 error, falling back to old_pull_img")
            return self.old_pull_img(pic)
        else:
            if S.img_caching:
                self._save_to_cache(pic, response.content)

        return response.content
    
    def old_pull_img(self, pic: Pic):
        # Check cache
        if S.img_caching:
            cached = self._pull_from_cache(pic)
            if cached:
                if S.request_msgs: print(f"[Cache] Hit for {pic.pano_id}")
                return cached
        
        # Increment usage count for current key
        self.usage_counts[self.key] += 1
        self._save_usage_counts()

        # Rotate key if usage exceeds max allowed
        if self.usage_counts[self.key] >= self.max_uses_per_key:
            print(f"[KEY ROTATION] {self.key} reached {self.max_uses_per_key} uses. Rotating...")
            self._rotate_key()

        # Add zoom level (FOV)
        zoom_to_fov = {0: 90, 1: 60, 2: 30}
        fov = zoom_to_fov[pic.zoom_lvl]

        # Parameters for API request
        pic_params = {
            'key': self.key,
            'return_error_code': True,
            'outdoor': True,
            'size': f"{S.img_width}x{S.img_height}",
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
        response = self._request(
            params = pic_params,
            context = "Pulling Image",
            url = 'https://maps.googleapis.com/maps/api/streetview?')
        
        # Close response, return content 
        content = response.content
        if S.img_caching:
            self._save_to_cache(pic, content)
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
        response = self._request(
            params=params,
            context="Pulling Metadata",
            url='https://maps.googleapis.com/maps/api/streetview/metadata?')
        
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

    def get_street_dir(self, pic):
        # Build request URL
        url = ("https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch"
                "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lng}!2d50!3m10"
                "!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2"
                "&callback=callbackfunc"
            ).format(lat=pic.lat, lng=pic.lng)
        
        # Build request header
        headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "*/*",
                "Referer": "https://maps.google.com/",
        }
        
        # Send request
        r = self._request(url=url, headers=headers, context="Pulling Street Dir")

        # Catch no response
        if r is None:
            return None
        
        # Strip JSON from payload text
        m = re.search(rf"{re.escape('callbackfunc')}\s*\(\s*(.*)\s*\)\s*;?\s*$", r.text, re.DOTALL)
        if not m:
            return None
        data = json.loads(m.group(1))

        # Check if we found anything
        if data == [[5, "generic", "Search returned no images."]]:
            return None

        # Pull out heading
        subset = data[1][5][0]
        raw_panos = subset[3][0]
        raw_panos = raw_panos[::-1]
        heading = float(raw_panos[0][2][2][0])
        return heading
    
    def _request(self, url, params=None, headers=None, context=""):
        """ Contains all the logic for making a request. """

        # Debugging
        if S.request_msgs: print(f"[Requests] {context}")
        for attempt in range(1, S.max_retries + 1):
            try:
                # Submit request
                response = requests.request(method="GET", url=url, params=params, headers=headers, timeout=10)
                response.raise_for_status()

                # Detect throttling. Return response
                if response.status_code == 403 or b"quota" in response.content.lower():
                    time.sleep(1.5 * attempt)
                    continue

                # Success
                if S.request_msgs: print(f"[Requests] Finished {context}")
                return response

            except (ReadTimeout, ConnectionError):
                print(f"[{context}] Timeout or connection error, retrying.")
            except HTTPError as e:
                print(f"[{context}] HTTPError {e.response.status_code}: {e.response.text[:50]}")
                if e.response.status_code in (429, 500, 503):
                    time.sleep(1.5 * attempt)
                    continue
                # Allows fallback for pulling images
                elif e.response.status_code == 400:
                    return response
                else:
                    break
            except RequestException as e:
                print(f"[{context}] RequestException: {e}")
                time.sleep(1.5 * attempt)
            except Exception as e:
                print(f"[{context}] Error: {e}")
                break
            
            # Sleep before retrying
            sleep_time = 1.5 * attempt + uniform(0, 0.5)
            time.sleep(sleep_time)

        print(f"[{context}] Failed after {attempt} tries.")
        return None
    
    """ Image caching """
    def _get_path(self, pic: Pic):
        if not pic.pano_id:
            return None
        fname = f"{pic.pano_id}_h{int(pic.heading or 0)}_z{pic.zoom_lvl}.jpg"
        return self.cache_dir / fname

    def _pull_from_cache(self, pic: Pic):
        path = self._get_path(pic)
        if path and path.exists():
            try:
                with open(path, "rb") as f:
                    return f.read()
            except Exception:
                print(f"[Cache] Failed to load {path}, deleting.")
                path.unlink(missing_ok=True)
        return None

    def _save_to_cache(self, pic: Pic, content: bytes):
        path = self._get_path(pic)
        if not path:
            return
        try:
            with open(path, "wb") as f:
                f.write(content)
        except Exception as e:
            print(f"[Cache] Failed to write {path}: {e}")

    """ Dealing with multiple API keys """
    def _load_usage_counts(self):
        if self.counter_path.exists():
            try:
                with open(self.counter_path, "r") as f:
                    self.usage_counts = json.load(f)
            except json.JSONDecodeError:
                print("[Warning] Could not parse api_calls.json, reinitializing...")
                self.usage_counts = {}
        else:
            self.usage_counts = {}

        # Ensure all keys have an entry
        for key in self.keys:
            if key not in self.usage_counts:
                self.usage_counts[key] = 0
    
    def _save_usage_counts(self):
        with open(self.counter_path, "w") as f:
            json.dump(self.usage_counts, f, indent=2)

    def _rotate_key(self):
        for i, key in enumerate(self.keys):
            if self.usage_counts.get(key, 0) < self.max_uses_per_key:
                self.key = key
                self.current_key_index = i
                return
        raise RuntimeError("All API keys have exceeded their limits.")