import math
import numpy as np
from shapely.geometry import Point, LineString
from pyproj import Transformer
import osmnx as ox
from settings import S
from src.utils.objects import Pic
from src.streetview.sv_requests import Reqs

EPS = 1e-6

class Move:
    def __init__(self, graph_cache, lat, lng, debug=False):
        """
        - Builds an OSMnx drive graph around (lat, lng)
        - Keeps a cleaned, projected edges GeoDataFrame
        """
        self.cache = graph_cache
        self.debug = debug
        self.reqs = Reqs()
        self.pic_cache = {}
        self.last_perp = None

        # Build graph around starting location (WGS84: lat/lng)
        self.G_osm = self.cache.get_graph(lat, lng)
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(self.G_osm)

        # Project edges to metric CRS (EPSG:3857) so distances are in meters
        edges_gdf = edges_gdf.to_crs(3857)
        self.edges_gdf = edges_gdf.copy()

        # Clean geometries: keep longest valid LineString for each edge
        def longest_valid_line(geom):
            if geom is None:
                return None
            if geom.geom_type == "MultiLineString":
                parts = [g for g in geom.geoms if g.length > 0 and g.is_valid]
                if not parts:
                    return None
                parts.sort(key=lambda g: g.length, reverse=True)
                return parts[0]
            if geom.geom_type == "LineString":
                if geom.length > 0 and geom.is_valid:
                    return geom
            return None

        self.edges_gdf["geom_line"] = self.edges_gdf.geometry.apply(longest_valid_line)
        self.edges_gdf = self.edges_gdf[self.edges_gdf["geom_line"].notnull()].copy()

        # Pre-compute length in meters
        self.edges_gdf["len_m"] = self.edges_gdf["geom_line"].apply(lambda g: g.length)

        # Ignore tiny stub edges that cause weird behavior at intersections
        self.edges_gdf = self.edges_gdf[self.edges_gdf["len_m"] >= 4.0].copy()

        # Spatial index for fast “nearby edges” queries
        try:
            self.edges_sindex = self.edges_gdf.sindex
        except Exception:
            self.edges_sindex = None

        # CRS transformers
        self.to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.to_wgs = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        # Hysteresis: remember last chosen edge index (tiny bias to keep going straight)
        self.last_edge_idx = None

    # -------------------------------------------------
    # PUBLIC MOVE
    # -------------------------------------------------
    def move(self, pic: Pic, backwards: bool = False):
        """
        Main entry point.

        1. Compute a movement vector in world coordinates based on the camera heading
        2. Choose the road segment (OSM edge) that best aligns with that movement.
        3. Move a fixed distance along that segment (clamped inside the segment).
        4. Ask GSV for metadata at the new location:
           - If get a *different* pano_id AND it lies in the intended movement
             direction (forward or backward), accept it.
           - Otherwise, scan further along the road network to find the nearest
             valid pano in that direction.
        """
        step_m = float(getattr(S, "dist", 10.0))

        # 1) Convert current pano position from lon/lat to metric (x,y)
        px, py = self.to_m.transform(pic.lng, pic.lat)
        pt_m = Point(px, py)  # original world-space position in meters

        # 2) Build movement vector:
        #    - heading_vec = where the camera is looking
        #    - movement_vec = direction we want to move in world coordinates
        #        * forwards:  movement_vec = heading_vec
        #        * backwards: movement_vec = -heading_vec
        heading_vec = self._heading_to_unitvec(pic.heading)
        movement_vec = heading_vec if not backwards else -heading_vec
        movement_vec = movement_vec / (np.linalg.norm(movement_vec) + EPS)

        # 3) Find nearby road edges
        candidates = self.get_nearby_edges(pt_m, radius=max(15, step_m * 1.6))
        if candidates.empty:
            if self.debug:
                print("No edges nearby")
            # If nowhere to go, return current pic 
            return pic

        # 4) Choose the best-aligned edge; projection gives us where we are on that edge
        best_idx, best_row, proj, chosen_dir = self.choose_best_edge(
            pt_m,
            movement_vec,
            candidates
        )
        if best_idx is None:
            return pic

        seg: LineString = best_row["geom_line"]
        seg_len = seg.length

        # Always interpret "forward/backwards" in camera terms
        signed_step = step_m if not backwards else -step_m

        # Desired new position along this edge in terms of arc-length:
        #   proj is current projection distance along seg (meters)
        #   signed_step moves us parallel to movement_vec
        target = proj + signed_step

        # Clamp movement inside this segment to avoid “falling off” intersections.
        target = max(0.0, min(seg_len, target))

        # 5) Compute the geometric new point along the segment
        new_pt_m = seg.interpolate(target)
        lng_geom, lat_geom = self.to_wgs.transform(new_pt_m.x, new_pt_m.y)

        # 6) Ask street view at that *geometric* location
        last_pano = getattr(pic, "pano_id", None)

        candidate_pic = self.get_metadata_for_point(lat_geom, lng_geom, pic.heading)

        # If we got a pano, we also check the *actual* pano location
        # (Google may snap it a few meters away).

        if candidate_pic and candidate_pic.pano_id != last_pano:
            # Recompute world direction from original position to pano location
            cx, cy = self.to_m.transform(candidate_pic.lng, candidate_pic.lat)
            pano_pt = Point(cx, cy)

            vec_to_pano = np.array(
                [pano_pt.x - pt_m.x, pano_pt.y - pt_m.y],
                dtype=float
            )
            dist_vec = np.linalg.norm(vec_to_pano)

            if dist_vec > 1e-3:
                vec_to_pano /= dist_vec
                # IMPORTANT:
                # movement_vec already encodes “forward or backward”.
                #   - For forward moves: movement_vec ~ +heading
                #   - For backward moves: movement_vec ~ -heading
                #
                # So in BOTH cases, we want dot(vec_to_pano, movement_vec) > 0
                # pano lies roughly in the intended movement direction.
                dot = float(np.dot(vec_to_pano, movement_vec))
            else:
                # If it's basically the same point, treat as “no directional info”
                dot = 0.0

            # Accept pano if it lies roughly in the movement direction (dot > 0)
            if dot > -.3:
                if self.debug:
                    print("[Move] Geometric step produced new pano directly "
                          f"(dot={dot:.3f}, dist={dist_vec:.1f}m).")
                # Accept this pano
                self.last_edge_idx = best_idx
                return candidate_pic
            else:
                # On direction mismatch, reject this pano and fall back to road scan.
                if self.debug:
                    print("[Move] Rejecting geometric pano: wrong direction "
                          f"(dot={dot:.3f}, dist={dist_vec:.1f}m).")

        # 7) If we reach here, either:
        #    - GSV gave same pano_id, or
        #    - No pano at all, or
        #    - Pano was in the wrong direction (dot <= 0).
        if self.debug:
            print("[Move] GSV returned same/none/wrong-direction pano. Scanning along road...")

        scanned_pic = self.search_for_valid_pano(
            start_edge_idx=best_idx,
            start_edge_row=best_row,
            start_proj=target,
            chosen_dir=chosen_dir,
            movement_vec=movement_vec,
            last_pano_id=last_pano,
            original_heading=pic.heading,
            pt_origin=pt_m,
            backwards=backwards
        )

        if scanned_pic is not None:
            self.last_edge_idx = best_idx
            return scanned_pic

        # If scanning also fails, we stay at the current pano.
        if self.debug:
            print("[Move] Road scan found no new pano. Staying at current pano.")
        return pic

    def get_nearby_edges(self, pt_m: Point, radius=20):
        """
        Return rows of self.edges_gdf whose geom_line is within `radius` meters
        of pt_m (shapely Point in metric CRS).
        """
        if self.edges_sindex is None:
            df = self.edges_gdf.copy()
            df["dist_m"] = df.geom_line.apply(lambda g: g.distance(pt_m))
            return df[df["dist_m"] <= radius].copy()

        minx, miny, maxx, maxy = pt_m.buffer(radius).bounds
        idxs = list(self.edges_sindex.intersection((minx, miny, maxx, maxy)))
        if not idxs:
            return self.edges_gdf.iloc[[]]

        df = self.edges_gdf.iloc[idxs].copy()
        df["dist_m"] = df.geom_line.apply(lambda g: g.distance(pt_m))
        return df[df["dist_m"] <= radius].copy()

    def choose_best_edge(self, pt_m, movement_vec, candidates):
        """
        Among nearby edges, choose the one that best aligns with movement_vec.

        For each candidate edge:
          - Project current point onto the edge → proj distance (meters along edge).
          - Compute local tangent at that projection (ahead - behind).
          - Consider both directions of that tangent (we don't trust OSM's u→v).
          - Compute alignment with movement_vec using dot products.
            The direction ( +tangent or -tangent ) giving the higher dot
            is considered the "forward" direction on that edge.
          - Score = alignment + tiny preference for longer segments
                   + small hysteresis bonus if we used this edge last time.

        Returns:
          - best_idx: the MultiIndex (u, v, key) of the chosen edge
          - best_row: the edges_gdf row
          - best_proj: projection distance along edge (meters)
          - best_dir: +1 if the chosen "forward direction" aligned with +tangent,
                      -1 if it aligned with -tangent.
        """
        best_score = -1e18
        best_row = None
        best_idx = None
        best_proj = None
        best_dir = +1

        for idx, row in candidates.iterrows():
            seg: LineString = row["geom_line"]
            if seg.length < 1e-6:
                continue

            # 1) Project current position onto this segment
            proj = seg.project(pt_m)

            # 2) Estimate local tangent direction using small epsilon ahead & behind
            eps = min(1.0, seg.length * 0.01)
            ahead = seg.interpolate(min(proj + eps, seg.length))
            behind = seg.interpolate(max(proj - eps, 0.0))

            tangent = np.array([ahead.x - behind.x, ahead.y - behind.y], dtype=float)
            n = np.linalg.norm(tangent)
            if n < 1e-9:
                continue
            # Calc unit tangent 
            tangent /= n 

            # For street side calculation
            self.last_perp = (-tangent[1], tangent[0])

            # 3) Evaluate alignment of BOTH pos/neg tangent with movement_vec
            align_pos = float(np.dot(tangent, movement_vec))
            align_neg = float(np.dot(-tangent, movement_vec))

            if align_pos >= align_neg:
                chosen_dir = +1
                align = align_pos
            else:
                chosen_dir = -1
                align = align_neg

            # 4) Score: alignment + small preference for length + tiny hysteresis
            score = align + 0.0005 * seg.length
            if idx == self.last_edge_idx:
                score += 0.05

            if score > best_score:
                best_score = score
                best_idx = idx
                best_row = row
                best_proj = proj
                best_dir = chosen_dir

        if self.debug and best_row is not None:
            seg = best_row["geom_line"]
            proj_pt = seg.interpolate(best_proj)
            print(" best_idx:", best_idx)
            print(" seg start:", seg.coords[0], "end:", seg.coords[-1], "len:", seg.length)
            print(" proj_dist:", best_proj, "proj_pt:", (proj_pt.x, proj_pt.y))
            print(" chosen_dir:", best_dir)

        return best_idx, best_row, best_proj, best_dir

    def search_for_valid_pano(
        self,
        start_edge_idx,
        start_edge_row,
        start_proj,
        chosen_dir,
        movement_vec,
        last_pano_id,
        original_heading,
        pt_origin,
        backwards,
        max_scan_m=40.0,
        step_m=4.0,
        max_hops=6
    ):
        """
        Scan along the road network in the intended direction looking for
        the nearest *different* pano_id that is still consistent with the
        movement_vec (which already encodes “forward or backward”).

        - Walks along the starting edge in steps of `step_m`.
        - If stepping beyond that edge, hops once onto the best-aligned
          neighboring edge (ignoring short stubs).
        - At each sampled location, we probe GSV using a small perpendicular
          jitter, and apply the same direction check as in the main move().

        Returns:
          Pic with updated pano_id/lat/lng if found, else None.
        """
        visited_edges = set()
        current_idx = start_edge_idx
        current_row = start_edge_row
        current_seg: LineString = current_row["geom_line"]
        current_proj = float(start_proj)
        traveled = 0.0
        hops = 0

        while traveled < max_scan_m and hops <= max_hops:
            # Step along current segment relative to its coordinate direction.
            # chosen_dir encodes which way along this edge we consider "forward"
            # w.r.t. movement_vec.
            dist_on_seg = current_proj + chosen_dir * step_m

            # If still within this segment, probe GSV here
            if 0.0 <= dist_on_seg <= current_seg.length:
                new_pic = self._probe_gsv_around_point(
                    seg=current_seg,
                    dist_on_seg=dist_on_seg,
                    last_pano_id=last_pano_id,
                    original_heading=original_heading,
                    pt_origin=pt_origin,
                    movement_vec=movement_vec,
                    backwards=backwards
                )
                if new_pic is not None:
                    if self.debug:
                        print(f"[Move] Found new pano after stepping ≈{traveled + step_m:.1f}m")
                    return new_pic

                # Advance along this segment and continue scanning
                current_proj = dist_on_seg
                traveled += step_m
                continue

            # Otherwise we are trying to step off the edge, causing intersection hop
            hops += 1
            try:
                u, v, key = current_idx
            except Exception:
                # Index not in (u, v, key) format
                return None

            # Pick which node we hit based on direction
            node_hit = v if dist_on_seg > current_seg.length else u

            # Get neighbor edges from this node
            next_edges = self._edges_from_node(node_hit)

            # Filter neighbors: not visited, not same edge, not stub
            candidates = []
            for eidx, erow in next_edges:
                if eidx == current_idx or eidx in visited_edges:
                    continue
                seg2: LineString = erow["geom_line"]
                if seg2.length < 4.0:
                    continue
                candidates.append((eidx, erow))

            if not candidates:
                # Nowhere else to go
                return None

            # Choose neighbor that best aligns with movement_vec (global direction)
            best_score = -1e18
            chosen = None
            node_xy = self._node_point_meters(node_hit)

            for eidx, erow in candidates:
                seg2: LineString = erow["geom_line"]
                coords = list(seg2.coords)
                start = coords[0]
                end = coords[-1]

                # Vector oriented away from node_hit along this edge
                start_dist = (start[0] - node_xy[0])**2 + (start[1] - node_xy[1])**2
                end_dist = (end[0] - node_xy[0])**2 + (end[1] - node_xy[1])**2
                if start_dist < end_dist:
                    vec = np.array([end[0] - start[0], end[1] - start[1]], float)
                    proj0 = 0.0
                    dir_sign = +1
                else:
                    vec = np.array([start[0] - end[0], start[1] - end[1]], float)
                    proj0 = seg2.length
                    dir_sign = -1

                n = np.linalg.norm(vec)
                if n < 1e-9:
                    continue
                vec /= n

                score = float(np.dot(vec, movement_vec)) + 0.001 * seg2.length
                if score > best_score:
                    best_score = score
                    chosen = (eidx, erow, proj0, dir_sign)

            if chosen is None:
                return None

            # Hop to that edge and continue scanning
            visited_edges.add(current_idx)
            current_idx, current_row, current_proj, chosen_dir = chosen
            current_seg = current_row["geom_line"]
            # Intersection hop is small; we don't increment traveled here
            continue

        return None

    def _probe_gsv_around_point(
        self,
        seg: LineString,
        dist_on_seg: float,
        last_pano_id,
        original_heading,
        pt_origin,
        movement_vec,
        backwards
    ):
        """
        Given a segment and a distance along it (metric arc-length),
        probe GSV at that point and small perpendicular offsets.

        Direction-aware behavior:
          - For each candidate pano returned by GSV:
              * Compute vector from original position → pano position.
              * Take dot product with movement_vec.
              * Because movement_vec already encodes “forward vs backward”,
                we simply require dot > 0 to accept the pano
                (it lies roughly in the intended movement direction).

        Returns:
          Pic with new pano_id if found, else None.
        """
        # Base point on segment in metric space
        base = seg.interpolate(dist_on_seg)

        # Local tangent & perpendicular
        eps = min(1.0, seg.length * 0.01)
        ahead = seg.interpolate(min(dist_on_seg + eps, seg.length))
        behind = seg.interpolate(max(dist_on_seg - eps, 0.0))

        tangent = np.array([ahead.x - behind.x, ahead.y - behind.y], float)
        n = np.linalg.norm(tangent)
        if n < 1e-9:
            tangent = np.array([1.0, 0.0], float)
            n = 1.0
        tangent /= n
        perp = np.array([-tangent[1], tangent[0]], float)

        # Try center, +0.5m, -0.5m across the road
        offsets = [0.0, 0.5, -0.5]

        for off in offsets:
            if abs(off) < 1e-9:
                cand = base
            else:
                cand = Point(base.x + off * perp[0], base.y + off * perp[1])

            # Call GSV metadata at this candidate metric point
            lng, lat = self.to_wgs.transform(cand.x, cand.y)
            tmp = self.get_metadata_for_point(lat, lng, original_heading)

            if tmp and tmp.pano_id == last_pano_id:
                # Same pano as original → not a move
                continue

            # Direction check: compare original → pano vector with movement_vec
            cx, cy = self.to_m.transform(tmp.lng, tmp.lat)
            pano_pt = Point(cx, cy)
            vec_to_pano = np.array(
                [pano_pt.x - pt_origin.x, pano_pt.y - pt_origin.y],
                dtype=float
            )
            dist_vec = np.linalg.norm(vec_to_pano)
            if dist_vec < 1e-3:
                # basically same position
                continue

            vec_to_pano /= dist_vec
            dot = float(np.dot(vec_to_pano, movement_vec))

            # IMPORTANT: movement_vec already encodes forward/backward.
            # So we simply require dot > 0 so that pano lies roughly in that direction.
            if dot <= 0.0:
                if self.debug:
                    print(f"[Move] Rejecting scanned pano (wrong direction, dot={dot:.3f})")
                continue

            # Found a new pano that is in the correct movement direction
            if self.debug:
                print(f"[Move] Accepting scanned pano (dot={dot:.3f}, dist≈{dist_vec:.1f}m)")
            return tmp

        return None

    def _edges_from_node(self, node):
        """
        Return a list of (edge_index, row) for edges incident to 'node'
        using the edges_gdf MultiIndex (u, v, key).
        """
        rows = []
        try:
            idx0 = self.edges_gdf.index.get_level_values(0)
            idx1 = self.edges_gdf.index.get_level_values(1)
            mask = (idx0 == node) | (idx1 == node)
            df = self.edges_gdf[mask]
            for idx, row in df.iterrows():
                rows.append((idx, row))
        except Exception:
            # Fallback: brute-force
            for idx, row in self.edges_gdf.iterrows():
                try:
                    u, v, _ = idx
                except Exception:
                    continue
                if u == node or v == node:
                    rows.append((idx, row))
        return rows

    def _node_point_meters(self, node):
        """
        Return (x,y) of graph node in meters (EPSG:3857).

        OSMnx stores nodes with 'x' (lon), 'y' (lat). We transform to metric
        coordinates to be consistent with edge geometries.
        """
        data = self.G_osm.nodes[node]
        lon = data.get("x", None)
        lat = data.get("y", None)
        if lon is None or lat is None:
            raise RuntimeError("Node missing coordinates")
        x, y = self.to_m.transform(lon, lat)
        return (x, y)

    def _heading_to_unitvec(self, heading_deg):
        """
        Convert a compass heading (0° = north, 90° = east, etc.)
        into a unit vector in metric XY coordinates.

        - We treat x as East, y as North (EPSG:3857 axes).
        - The expression (90 - heading_deg) converts compass bearing
          into standard math angle measured from +x axis ccw.
        """
        rad = math.radians((90 - heading_deg) % 360)
        return np.array([math.cos(rad), math.sin(rad)], float)
    

    def get_metadata_for_point(self, lat, lon, heading):
        """
        Unified metadata lookup:
        - Checks cache first.
        - Calls metadata only once per coordinate.
        - Returns a Pic containing pano_id, coords, etc.
        """

        key = self._coord_key(lat, lon)

        # Cache hit
        if key in self.pic_cache:
            pic = self.pic_cache[key]
            cached = Pic(heading, pic.lat, pic.lng)
            cached.pano_id = pic.pano_id
            cached.date = pic.date
            cached.zoom_lvl = 0
            return cached

        # Cache miss (call metadata)
        tmp = Pic(heading, lat, lon)
        tmp.zoom_lvl = 0
        tmp.pano_id = None

        ok = self.reqs.pull_pano_info(tmp)

        if ok and tmp.pano_id:
            # store in cache
            self.pic_cache[key] = tmp
            return tmp

        # No pano found
        return None

    def _coord_key(self, lat, lon, precision=7):
        """
        Produces a stable hashable key for coordinates.
        Rounding avoids floating drift and matches GSV precision.
        """
        return (round(lat, precision), round(lon, precision))
    
    def calc_rd_vectors(self, pic: Pic):
        """
        Computes tangent and perpendicular road vectors at the initial agent
        position (x, y). This must be called before any movement so detections
        can be classified immediately.
        """
        # Current pos to metric
        px, py = self.to_m.transform(pic.lng, pic.lat)
        pt_m = Point(px, py)

        # Compute a movement vec bc we have to
        movement_vec = self._heading_to_unitvec(pic.heading)

        # 3) Find nearby road edges
        candidates = self.get_nearby_edges(pt_m, radius=15)

        # Basically dry fire choose best edge so it will update last tangent and last perp.
        self.choose_best_edge(pt_m, movement_vec, candidates)

    def get_road_vec(self):
        """ Used to determine street side of a detection. """
        return self.last_perp