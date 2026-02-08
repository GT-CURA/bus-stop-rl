from src.utils.objects import Stop
from pyproj import CRS, Transformer

class RoadContext:
    def __init__(self):
        self.segment = None
        self.tangent = None
        self.perp = None
        self.stop: Stop = None
        self.to_local = None
        self.to_global = None
        self.local_crs = None

    def set_context(self, stop: Stop):
        self.stop = stop
        self.make_transforrmer(stop.og_lat, stop.og_lng)
    
    def make_transforrmer(self, lat, lng):
        wgs84 = CRS.from_epsg(4326)
        self.local_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat} +lon_0={lng} +datum=WGS84 +units=m"
        )
        self.to_local = Transformer.from_crs(wgs84, self.local_crs, always_xy=True)
        self.to_global = Transformer.from_crs(self.local_crs, wgs84, always_xy=True)