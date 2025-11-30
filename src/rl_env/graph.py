from collections import deque
import numpy as np
from src.utils.objects import Hypothesis

class Node:
    def __init__(self, lat, lng):
        # Location
        self.lat = lat
        self.lng = lng

        # Neighbor pano IDs
        self.neighbors = set()

        # Detections and scorecard
        self.best_conf = 0.0
        self.best_bearing = None
        self.scores = {"shelter": 0.0, "sign": 0.0, "trash can": 0.0, "seating": 0.0}

        # Detections (class in objects module) 
        self.detections = []

        # Visit counter
        self.visits = 0

class Graph:

    def __init__(self, debug=False):
        self.graph = {}
        self.prev_node_id = None
        self.hypotheses = []
        self.prev_dist_to_best = None
        self.prev_det_bearing_err = None
        self.debug = debug
        self.conf_threshold = .4
        self.prev_coord_dist = None

    def add_node(self, pic, add_neighbors = True):
        """ Adds and updates nodes. """
        # Add node to graph if not there
        if pic.pano_id not in self.graph:
            self.graph[pic.pano_id] = Node(pic.lat, pic.lng)
            self.graph[pic.pano_id].pano_id = pic.pano_id

        # Get node, increase visits and set heading
        node = self.graph[pic.pano_id]
        node.visits += 1
        node.current_heading = pic.heading

        # Add bidirectional neighbor edge
        if add_neighbors:
            if self.prev_node_id is not None and self.prev_node_id != pic.pano_id:
                node.neighbors.add(self.prev_node_id)
                self.graph[self.prev_node_id].neighbors.add(pic.pano_id)

        self.prev_node_id = pic.pano_id
        return node


    def calc_graph_rwd(self, current_pano_id):
        """
        Potential-based shaping reward using BFS distance to the single
        best pano node (highest best_conf). Works even before hypotheses exist.
        """

        if not self.graph:
            return 0.0

        # Find pano with max best_conf
        best_id = max(self.graph.keys(), key=lambda nid: self.graph[nid].best_conf)
        if self.graph[best_id].best_conf == 0:
            self.prev_dist_to_best = None
            return 0.0

        # Find shortest distance to that pano
        dist = self.shortest_distance(current_pano_id, [best_id])
        if dist is None:
            return 0.0

        # Initialize
        if self.prev_dist_to_best is None:
            self.prev_dist_to_best = dist
            return 0.0

        # Reward is difference between 
        reward = self.prev_dist_to_best - dist
        self.prev_dist_to_best = dist
        return reward
    
    def shortest_distance(self, start_id, target_ids):
        """ Finds shortest distance between two nodes. """
        # If already at target, return 0 
        if start_id in target_ids:
            return 0

        # Build queue of (ids, dists)
        visited = set()
        queue = deque([(start_id, 0)])

        while queue:
            # Get latest node, mark as visited 
            nid, dist = queue.popleft()
            visited.add(nid)

            # Visit neighbors, adding to queue 
            for nb in self.graph[nid].neighbors:
                if nb in target_ids:
                    return dist + 1
                if nb not in visited:
                    queue.append((nb, dist + 1))

        return None


    def calc_direction_rwd(self, curr_node, curr_heading):
        """
        Direction reward: 
        1. If the current frame produced a detection, use its bearing immediately.
        2. If no detections inthis frame, , use hypothesis.best_bearing.
        """

        def wrap(a):
            return ((a + 180) % 360) - 180

        # Get latest detection
        latest_det = None
        if len(curr_node.detections) > 0:
            latest_det = curr_node.detections[-1]
        
        # Default to using latest detection
        if latest_det is not None:
            desired = latest_det.bearing
            curr_err = abs(wrap(curr_heading - desired))

            if self.prev_det_bearing_err is None:
                self.prev_det_bearing_err = curr_err
                return 0.0

            reward = (self.prev_det_bearing_err - curr_err) / 180.0
            self.prev_det_bearing_err = curr_err

            # Also update hypothesis bearing, if one exists
            if self.hypotheses:
                best = max(self.hypotheses, key=lambda h: h.score)
                best.best_bearing = desired

            return reward

        # Fall back to using hypotheses 
        if self.hypotheses:
            best = max(self.hypotheses, key=lambda h: h.score)
            if getattr(best, "best_bearing", None) is None:
                return 0.0

            desired = best.best_bearing
            curr_err = abs(wrap(curr_heading - desired))

            if getattr(best, "prev_bearing_err", None) is None:
                best.prev_bearing_err = curr_err
                return 0.0

            reward = (best.prev_bearing_err - curr_err) / 180.0
            best.prev_bearing_err = curr_err
            return reward

        # No directional signal
        return 0.0

    def calc_coord_rwd(self, pano_x, pano_y):
        """
        Encourgages movement towards an estimated stop coordinate. 
        """

        # Only use hypotheses with a triangulated position
        triangulated = [h for h in self.hypotheses if h.triangulated_pos is not None]
        if not triangulated:
            return 0.0

        # Get best hypothesis 
        best = max(triangulated, key=lambda h: h.score)
        best_x, best_y = best.triangulated_pos

        # Calculate distance between pano and best hyp
        dist = np.linalg.norm([pano_x - best_x, pano_y - best_y])

        # Init previous distance
        if self.prev_coord_dist is None:
            self.prev_coord_dist = dist
            return 0.0

        # Calc reward
        reward = (self.prev_coord_dist - dist) / 20.0
        self.prev_coord_dist = dist
        return reward
    
    def triangulate(self, rays):
        """
        rays: list of dicts with {x, y, bearing}
        Returns (x,y) triangulated intersection or None if rays are parallel.
        """
        # Return if no rays
        if len(rays) < 2:
            return None

        pts = []
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                r1, r2 = rays[i], rays[j]

                # Convert first cords to ray 
                x1, y1 = r1["x"], r1["y"]
                t1 = np.radians(r1["bearing"])
                d1 = np.array([np.cos(t1), np.sin(t1)])

                # Convert second coords to ray 
                x2, y2 = r2["x"], r2["y"]
                t2 = np.radians(r2["bearing"])
                d2 = np.array([np.cos(t2), np.sin(t2)])

                A = np.array([[d1[0], -d2[0]],
                              [d1[1], -d2[1]]])
                b = np.array([x2 - x1, y2 - y1])

                # Skip if rays are parallel 
                if abs(np.linalg.det(A)) < 1e-6:
                    continue

                t_sol = np.linalg.solve(A, b)
                t1_sol = t_sol[0]

                px = x1 + t1_sol * d1[0]
                py = y1 + t1_sol * d1[1]
                pts.append((px, py))

        if not pts:
            return None

        # Return points 
        pts = np.array(pts)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def match_hypothesis(self, hyp, det):
        """
        Checks if a given detection matches an existing hypothesis.
        """
        # Type match
        if hyp.label != det.label:
            return False

        # Ensure bearings match (take mean of bearings)
        mean_bearing = np.mean([o.bearing for o in hyp.observations])
        diff = abs(((mean_bearing - det.bearing + 180) % 360) - 180)
        if diff > 60:
            return False

        return True

    def update_hypotheses(self, node, step):
        """ Evaluate detections from this frame and update hypotheses.
        """

        # Only use detections from this frame
        frame_dets = [d for d in node.detections if d.timestamp == step]
        if not frame_dets:
            return

        for det in frame_dets:
            
            # Ensure deteection's conf meets threshold 
            if det.primary_conf < self.conf_threshold:
                continue

            merged = False

            # Try merging into existing hypotheses
            for hyp in self.hypotheses:
                if self.match_hypothesis(hyp, det):
                    hyp.observations.append(det)

                    # Triangulate from all hyp's observations
                    rays = [{"x": o.local_x, "y": o.local_y, "bearing": o.bearing}
                            for o in hyp.observations]
                    tri = self.triangulate(rays)
                    if tri is not None:
                        hyp.triangulated_pos = tri

                        # Reset coordinate reward memory
                        self.prev_coord_dist = None

                    # Update score (includes box size)
                    hyp.score = max(hyp.score, det.primary_conf * (det.box_sz + 1e-6))
                    hyp.last_seen = step
                    merged = True
                    break

            if merged:
                continue

            # Create new hypothesis
            new_score = det.primary_conf * (det.box_sz + 1e-6)
            new_hyp = Hypothesis(
                observations=[det],
                triangulated_pos=None,
                score=new_score,
                label=det.label,
                last_seen=step,
                best_bearing=det.bearing,
                prev_bearing_err=None
            )
            self.hypotheses.append(new_hyp)

        # Keep 3 higest scoring hyps
        self.hypotheses = sorted(self.hypotheses, key=lambda h: -h.score)[:3]


    def get_features(self, node, curr_lat, curr_lng, curr_heading):
        """ Producesa feature vector providing info relevant to spatial rewards system. """
        # 1. Calculate direction error 
        def wrap(a):
            return ((a + 180) % 360) - 180

        # Get latest detection
        latest_det = node.detections[-1] if node.detections else None

        # Set desired bearing to latest detection, or fall back to hypothessi 
        desired_bearing = None
        if latest_det is not None:
            desired_bearing = latest_det.bearing
        else:
            if self.hypotheses:
                best_h = max(self.hypotheses, key=lambda h: h.score)
                if getattr(best_h, "best_bearing", None) is not None:
                    desired_bearing = best_h.best_bearing

        # Calc heading error, break into sin cos
        if desired_bearing is not None:
            bearing_err = wrap(curr_heading - desired_bearing)
            err_sin = np.sin(np.radians(bearing_err))
            err_cos = np.cos(np.radians(bearing_err))
        else:
            err_sin, err_cos = 0.0, 0.0

        # 2. Coordinate reward features
        # Find hypotheses with triangulated pos 
        triangulated = [h for h in self.hypotheses if h.triangulated_pos is not None]

        if triangulated:
            # Get best hyp
            best_hyp = max(triangulated, key=lambda h: h.score)

            # Get dist to best hyp, scale it 
            best_x, best_y = best_hyp.triangulated_pos
            coord_dist = np.linalg.norm([curr_lat - best_x, curr_lng - best_y])
            coord_dist_scaled = np.tanh(coord_dist / 50)

            # Calc change in distance
            if self.prev_coord_dist is not None:
                delta_coord = self.prev_coord_dist - coord_dist
            else:
                delta_coord = 0.0

            # Tell agent if we have a triangulated hyp to pursue
            has_triangulated = 1.0
        else:
            coord_dist_scaled = 0.0
            delta_coord = 0.0
            has_triangulated = 0.0

        # 3. Graph reward features
        if self.graph:
            # GEt node with best confidence
            best_id = max(self.graph.keys(), key=lambda nid: self.graph[nid].best_conf)
            best_conf = self.graph[best_id].best_conf

            # Find distance to best node 
            bfs_dist = self.shortest_distance(node.pano_id, [best_id])
            bfs_scaled = np.tanh((bfs_dist or 0) / 10)

            # Calc chance in distance to best node
            if self.prev_dist_to_best is not None and bfs_dist is not None:
                delta_bfs = self.prev_dist_to_best - bfs_dist
            else:
                delta_bfs = 0.0

        else:
            best_conf = 0.0
            bfs_scaled = 0.0
            delta_bfs = 0.0

        # 4. Local graph features
        visit_scaled = min(node.visits / 10, 1)
        degree_scaled = min(len(node.neighbors) / 10, 1)

        # 5. Build the vector 
        return np.array([
            err_sin,
            err_cos,
            coord_dist_scaled,
            delta_coord,
            has_triangulated,
            bfs_scaled,
            delta_bfs,
            best_conf,
            visit_scaled,
            degree_scaled
        ], dtype=np.float32)
