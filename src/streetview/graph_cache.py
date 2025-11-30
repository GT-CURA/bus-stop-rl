import os
import osmnx as ox
from settings import S 
class GraphCache:
    def __init__(self, tile_size=0.03):
        """
        tile_size: degrees (0.02 is about 2.2 km)
        """
        self.tile_size = tile_size
        self.cache_dir = S.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_cache = {}

    def _tile_key(self, lat, lon):
        return (round(lat / self.tile_size), round(lon / self.tile_size))

    def get_graph(self, lat, lon, radius=800):
        """ Returns an OSMnx graph for the tile containing stop pos """
        key = self._tile_key(lat, lon)
        fname = os.path.join(self.cache_dir, f"tile_{key[0]}_{key[1]}.graphml")

        # Check memory cache
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Check disk cache
        if os.path.exists(fname):
            G = ox.load_graphml(fname)
            self.memory_cache[key] = G
            return G

        # Fetch from overpass 
        print(f"[Graph Cache] Downloading new tile {key}…")
        G = ox.graph_from_point((lat, lon), dist=radius, network_type="drive")
        ox.save_graphml(G, fname)
        self.memory_cache[key] = G
        return G
