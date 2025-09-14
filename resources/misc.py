import numpy as np 
from settings import S
from cv2 import resize
from math import radians, sin, cos, sqrt, atan2

class Misc:
    def preprocess_img(img):
        """ Resizes, normalizes, and transposes images. """
        img = resize(img, S.img_size).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img 

    def haversine(lat1, lon1, lat2, lon2):
        """ Implementation of the haversine formula to obtain distance from initial to new cords. """
        R = 6371000
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))
        
    def announce(instance, key, reward):
        """ Print stop info and action to console at each step. """
        name = instance.stop.place_name
        lat = instance.stop.og_lat
        lon = instance.stop.og_lng
        num_views = len(instance.found_viewpoints)
        print(f"[Step {instance.steps}] Action: '{key}' | Reward: {reward:.3f} | Viewpoints: {num_views} | Since Found: {instance.steps_since_found}")
        print(f"Stop: {name} ({lat}, {lon})")