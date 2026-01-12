from pathlib import Path
from requests.exceptions import RequestException, ReadTimeout, HTTPError, ConnectionError
import time
import json
from requests.exceptions import RequestException
from random import uniform
import requests
from settings import S
from src.utils.objects import Pic

class Reqs:
    def __init__(self):
        self.max_uses_per_key = 9000
        self.keys = []
        self.current_key_index = 0
        self.counter_path = Path(f"{S.log_dir}/api_calls.json")

        # Load keys
        with open(S.key_path, "r") as f:
            self.keys = [k.strip() for k in f.read().split(",") if k.strip()]

        if not self.keys:
            raise ValueError("No API keys provided.")
        
        # Load call counts
        self._load_usage_counts()
        self._rotate_key()

    def pull_image(self, pic: Pic):
        # Get pano ID if we don't have it
        if not pic.pano_id:
            self.pull_pano_info(pic)

        url = f"https://streetviewpixels-pa.googleapis.com/v1/thumbnail?cb_client=maps_sv.tactile&w=640&h=640&panoid={pic.pano_id}&yaw={pic.heading}&pitch=0.00"
        response = self._request(url, context="Pulling Thumbnail")
        # If API returns error image or no content
        if response.status_code == 400:
            print("[Requests] Got 400 error, falling back to old_pull_img")
            return self.old_pull_img(pic)
        
        return response.content
    
    def old_pull_img(self, pic: Pic):
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

        # Use either coord location or pano ID (if exists)
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
        
        if pano_location["lng"] is None:
            print(response.content)
            return False 
        
        # Fetch the coordinates from the json response and store them in the POI
        pano_location = response.json().get("location")
        pic.lng = pano_location["lng"]
        pic.lat = pano_location["lat"]
        pic.pano_id = response.json().get("pano_id")
        pic.date = response.json().get("date")
        response.close()
        return True

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