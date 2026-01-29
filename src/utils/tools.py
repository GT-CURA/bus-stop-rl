from math import radians, sin, cos, sqrt, atan2
import numpy as np

def haversine(lat1, lng1, lat2, lng2):
    """ Implementation of the haversine formula to obtain distance from initial to new cords. """
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def localize_coords(lat, lng, initial_lat, initial_lng):
    """ Make cords relative to origin (starting position for this stop) """
    R = 6371000
    dlat = np.radians(lat - initial_lat)
    dlon = np.radians(lng - initial_lng)
    x = R * dlon * np.cos(np.radians(initial_lat))
    y = R * dlat
    return x, y

def globalize_coords(x, y, initial_lat, initial_lng):
    """ Reverts coords from local plane to global"""
    R = 6371000.0
    dlat = y / R
    dlon = x / (R * np.cos(np.radians(initial_lat)))
    lat = initial_lat + np.degrees(dlat)
    lng = initial_lng + np.degrees(dlon)
    return lat, lng
