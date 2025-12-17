import os
import json
import osmnx as ox
from shapely.geometry import box, Point
from rtree import index
from pyproj import Transformer
from settings import S
from time import time

class GraphCache:
    """ Handles pulling and caching graphs from OSMNX. """

    def __init__(self, radius=900):
        self.radius = radius

        # Make cache directory
        self.cache_dir = S.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # In-memory storage
        self.graphs = {}
        self.metadata = {}

        # Spatial index over graph bboxes
        self.sindex = index.Index()

        # CRS transformer
        self.to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # Load existing graphs from disk
        self._load_from_disk()

    def get_graph(self, lat, lon):
        """ Return a graph, downloading a new one if none exists. """

        # Transform pt
        px, py = self.to_m.transform(lon, lat)
        pt = Point(px, py)

        # Use rtree spatial index to find overlapping BB
        for gid in self.sindex.intersection((px, py, px, py)):
            key = f"graph_{gid}"
            meta = self.metadata[key]
            if meta["bbox"].contains(pt):
                return self.graphs[key]

        # Download new graph if none found
        return self._download_graph(lat, lon)

    def _download_graph(self, lat, lon):
        gid = f"graph_{len(self.graphs)}"
        print(f"[Graph Cache] Downloading graph centered at ({lat:.6f}, {lon:.6f})")

        G = ox.graph_from_point(
            (lat, lon),
            dist=self.radius,
            network_type="drive"
        )

        # Convert to GeoDataFrame to get bounds
        _, edges = ox.graph_to_gdfs(G)
        edges = edges.to_crs(3857)
        minx, miny, maxx, maxy = edges.total_bounds
        bbox = box(minx, miny, maxx, maxy)

        # Build paths for metadata 
        graph_path = os.path.join(self.cache_dir, f"{gid}.graphml")
        meta_path = os.path.join(self.cache_dir, f"{gid}.json")

        # Save graph and metadatta
        ox.save_graphml(G, graph_path)
        with open(meta_path, "w") as f:
            json.dump({
                "center": [lat, lon],
                "radius": self.radius,
                "bbox": [minx, miny, maxx, maxy]
            }, f)

        # Register in memory
        self.graphs[gid] = G
        self.metadata[gid] = {
            "center": (lat, lon),
            "radius": self.radius,
            "bbox": bbox
        }

        self.sindex.insert(len(self.graphs) - 1, (minx, miny, maxx, maxy))

        return G

    def _load_from_disk(self):
        """
        Load all cached graphs + metadata into memory and spatial index.
        """
        # Iterate through each file in cache dir
        for fname in os.listdir(self.cache_dir):
            if not fname.endswith(".json"):
                continue
            
            # Get metadata and graph files
            gid = fname.replace(".json", "")
            meta_path = os.path.join(self.cache_dir, fname)
            graph_path = os.path.join(self.cache_dir, f"{gid}.graphml")

            if not os.path.exists(graph_path):
                continue
            
            # Load metadata and graph
            with open(meta_path, "r") as f:
                meta = json.load(f)
            G = ox.load_graphml(graph_path)

            # Get bounding box from metadata
            minx, miny, maxx, maxy = meta["bbox"]
            bbox = box(minx, miny, maxx, maxy)

            # Put graph and metadata in memory
            idx = len(self.graphs)
            self.graphs[gid] = G
            self.metadata[gid] = {
                "center": tuple(meta["center"]),
                "radius": meta["radius"],
                "bbox": bbox
            }

            # Insert into rtree
            self.sindex.insert(idx, (minx, miny, maxx, maxy))