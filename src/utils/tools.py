from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lng1, lat2, lng2):
    """ Implementation of the haversine formula to obtain distance from initial to new cords. """
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))