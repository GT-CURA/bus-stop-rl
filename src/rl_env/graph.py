from collections import deque
import numpy as np
from src.utils.objects import Hypothesis, Detection, Pic, Node
from shapely.geometry import Point
from src.utils.context import RoadContext

class Graph:
    def __init__(self, context: RoadContext, debug=False):
        self.graph = {}
        self.prev_node_id = None
        self.hypotheses = []
        self.prev_dist_to_best = None
        self.prev_det_bearing_err = None
        self.debug = debug
        self.conf_threshold = .4
        self.prev_coord_dist = None
        self.prev_coord_hyp = None
        self.last_dir_err = (0.0, 0.0)
        self.last_coord_dist_scaled = 0.0
        self.last_has_triangulated = 0.0
        self.context = context

    def add_node(self, pic: Pic, add_neighbors = True):
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
        Calculates distance to nodes with best observations. 
        """
        # Log for get_features
        self.last_bfs_scaled = 0.0

        # Return nothing if no hypotheses exist 
        if not self.hypotheses:
            return 0.0

        # Find best hypothesis
        best_hyp = max(self.hypotheses, key=lambda h: h.score)

        # Find pano_ids related to this hyp
        pano_ids = {det.pano_id for det in best_hyp.observations}
        if not pano_ids:
            return 0.0

        # Shortest distance to panos 
        dist = self.shortest_distance(current_pano_id, pano_ids)
        if dist is None:
            return 0.0

        # Update previous dist to best 
        if self.prev_dist_to_best is None:
            self.prev_dist_to_best = dist
            return 0.0

        # Calc reward 
        reward = self.prev_dist_to_best - dist
        self.prev_dist_to_best = dist

        # If dist, set for get_features
        self.last_bfs_scaled = np.tanh(dist / 10.0)
        return reward
    
    def shortest_distance(self, start_id, target_ids):
        """ Finds shortest distance between start and target nodes. """
        # If already at target, return 0 
        if start_id in target_ids:
            return 0

        visited = set()
        
        # Create queue with each target ID having a dist of zero 
        queue = deque([(tid, 0) for tid in target_ids if tid in self.graph])

        # Mark all targets as visited
        for tid in target_ids:
            visited.add(tid)

        # Pop queue until hit target ID 
        while queue:
            nid, dist = queue.popleft()

            # Explore neighbors
            for nb in self.graph[nid].neighbors:
                if nb == start_id:
                    return dist + 1
                
                # Mark as visited
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))

        return None

    def calc_direction_rwd(self, curr_node: Node, pic: Pic):
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
            err = wrap(pic.heading - desired)

            # Store bearing error for get_features
            self.last_dir_err = (
                np.sin(np.radians(err)),
                np.cos(np.radians(err))
            )

            # Take absolute value for reward
            curr_err = abs(err)
            if self.prev_det_bearing_err is None:
                self.prev_det_bearing_err = curr_err
                return 0.0
            
            # Calc reward 
            reward = (self.prev_det_bearing_err - curr_err) / 180.0
            self.prev_det_bearing_err = curr_err

            # Also update hypothesis bearing, if one exists
            if self.hypotheses:
                best = max(self.hypotheses, key=lambda h: h.score)
                best.best_bearing = desired

            return reward

        # Fall back to using hypotheses 
        if self.hypotheses:
            # Get best hypothesis
            best = max(self.hypotheses, key=lambda h: h.score)

            # Bearing from current pano to triangulated stop
            if best.triangulated_pos is not None:
                pano_x, pano_y = self.context.to_local.transform(pic.lng, pic.lat)
                desired = self.bearing_from(pano_x, pano_y, best.triangulated_pos)
            elif getattr(best, "best_bearing", None) is not None:
                # Fallback: use stored bearing if no triangulation yet
                desired = best.best_bearing
            else:
                return 0.0

            # Store error for get_features
            err = wrap(pic.heading - desired)
            self.last_dir_err = (
                np.sin(np.radians(err)),
                np.cos(np.radians(err))
            )

            # Update prev bearing error
            curr_err = abs(err)
            if getattr(best, "prev_bearing_err", None) is None:
                best.prev_bearing_err = curr_err
                return 0.0

            # Calc reward 
            reward = (best.prev_bearing_err - curr_err) / 180.0
            best.prev_bearing_err = curr_err
            return reward

        # No directional signal
        return 0.0

    def calc_coord_rwd(self, pic: Pic):
        """ Reward agent for stepping towards an estimated coord for bus stop. """
        pano_x, pano_y = self.context.to_local.transform(pic.lng, pic.lat)

        # Only use hypotheses with a triangulated position
        triangulated = [h for h in self.hypotheses if h.triangulated_pos is not None]
        if not triangulated:
            self.prev_coord_hyp = None
            self.prev_coord_dist = None
            self.last_coord_dist_scaled = 0.0
            self.last_has_triangulated = 0.0
            return 0.0

        # GEt best hyp
        best = max(triangulated, key=lambda h: h.score)
        best_x, best_y = best.triangulated_pos

        # Current distance from pano to best hypothesis
        dist = np.linalg.norm([pano_x - best_x, pano_y - best_y])

        # Log distance for get_features
        self.last_coord_dist_scaled = np.tanh(dist / 50.0)
        self.last_has_triangulated = 1.0

        # If we just switched to a different best hypothesis, (re)initialize
        if self.prev_coord_hyp is not best:
            self.prev_coord_hyp = best
            self.prev_coord_dist = dist
            return 0.0

        # If first time tracking this hypothesis
        if self.prev_coord_dist is None:
            self.prev_coord_dist = dist
            return 0.0

        # Positive when moving closer, negative when moving away
        reward = (self.prev_coord_dist - dist) / 10.0 
        self.prev_coord_dist = dist
        return reward
    
    def triangulate(self, rays):
        """ Triangulate pos of bus stop using least squares """

        if len(rays) < 2:
            return None

        A = []
        b = []

        for r in rays:
            x0, y0 = r["x"], r["y"]
            theta = np.radians(r["bearing"])

            # Unit direction of ray
            d = np.array([np.sin(theta), np.cos(theta)])

            # Normal vector to ray
            n = np.array([-d[1], d[0]])

            # Line constraint
            A.append(n)
            b.append(n @ np.array([x0, y0]))

        A = np.vstack(A)
        b = np.array(b)

        # Bearrings parallel
        ATA = A.T @ A
        if np.linalg.cond(ATA) > 1e6:
            return None

        # Solve least squares
        xy = np.linalg.solve(ATA, A.T @ b)
        return float(xy[0]), float(xy[1])

    def bearing_from(self, x1, y1, target_xy):
        x2, y2 = target_xy

        dx = x2 - x1
        dy = y2 - y1

        # Convert so that 0 degrees is north
        angle = np.degrees(np.arctan2(dy, dx))
        bearing = (90.0 - angle) % 360.0

        return bearing

    def angular_diff(self, a, b):
        """ Returns smallest angular difference between two bearings. """
        diff = (a - b + 180) % 360 - 180
        return abs(diff)
    
    def update_hypotheses(self, node: Node, step: int, action: str):
        """ Evaluate detections from this frame and update hypotheses. """

        # Only use detections from this frame
        frame_dets = [d for d in node.detections if d.timestamp == step]
        if not frame_dets:
            return

        for det in frame_dets:

            # Ensure detection's conf meets threshold
            if det.primary_conf < self.conf_threshold:
                continue

            merged = False

            # Try merging into existing hypotheses
            for hyp in self.hypotheses:
                if self.match_hypothesis(hyp, det):
                    hyp.observations.append(det)

                    # Organize observations into rays
                    rays = [
                        {"x": o.local_x, "y": o.local_y, "bearing": o.bearing}
                        for o in hyp.observations
                    ]

                    # Triangulate rays 
                    tri = self.triangulate(rays)
                    if tri is not None:
                        seg = self.context.segment
                        last_perp = self.context.perp

                        # Project triangulated point onto road
                        pt = Point(tri)
                        proj = seg.project(pt)
                        on_road = seg.interpolate(proj)

                        # Calculate which side the triangulation is on
                        vec_to_tri = np.array([tri[0] - on_road.x, tri[1] - on_road.y])
                        distance_from_road = np.linalg.norm(vec_to_tri)
                        
                        if distance_from_road > 0.1:
                            # Normalize and scale to curb
                            direction = vec_to_tri / distance_from_road
                            curb_offset = 7.5
                            
                            # Move toward the triangulated side
                            x = on_road.x + curb_offset * direction[0]
                            y = on_road.y + curb_offset * direction[1]
                        else:
                            # Fallback: use perpendicular with detection side
                            side_sign = 1.0 if hyp.side == "left" else -1.0
                            curb_offset = 7.5
                            x = on_road.x + side_sign * curb_offset * last_perp[0]
                            y = on_road.y + side_sign * curb_offset * last_perp[1]
                        
                        hyp.triangulated_pos = (x, y)

                    # Update score (includes box size)
                    hyp.score = max(
                        hyp.score,
                        det.primary_conf
                    )
                    hyp.last_seen = step
                    merged = True
                    break

            if merged:
                continue

            # Create new hypothesis. Scale by box sz unless zooming
            if action == "Zoom":
                new_score = det.primary_conf * .01
            else:
                new_score = det.primary_conf * (det.box_sz + .01)
            new_hyp = Hypothesis(
                observations=[det],
                triangulated_pos=None,
                score=new_score,
                label=det.label,
                last_seen=step,
                best_bearing=det.bearing,
                prev_bearing_err=None,
                side=det.side
            )
            self.hypotheses.append(new_hyp)

        # Keep top 3 hypotheses by score
        self.hypotheses = sorted(self.hypotheses, key=lambda h: -h.score)[:3]

    def match_hypothesis(self, hyp: Hypothesis, det: Detection):
        """
        Checks if a given detection matches an existing hypothesis.
        """
        # Type match
        if hyp.label != det.label:
            return False

        # Ensure same side of street (logic handled in Streetview lol) 
        if hyp.side != det.side:
            return False
        
        return True

    def get_features(self, node: Node):
        """ Produces a feature vector providing info relevant to spatial rewards system. """

        # 1. Direction error features
        dir_sin, dir_cos = self.last_dir_err if hasattr(self, "last_dir_err") else (0.0, 0.0)

        # 2. Coordinate reward features
        coord_dist_scaled = getattr(self, "last_coord_dist_scaled", 0.0)
        has_triangulated = getattr(self, "last_has_triangulated", 0.0)

        # 3. Graph reward features
        bfs_scaled = getattr(self, "last_bfs_scaled", 0.0)

        # Local graph structure
        visit_scaled = min(node.visits / 10.0, 1.0)
        degree_scaled = min(len(node.neighbors) / 10.0, 1.0)

        # Node confidence can help measure usefulness of this location
        best_conf = node.best_conf

        # Final PPO-compatible vector
        return np.array([
            dir_sin,
            dir_cos,
            coord_dist_scaled,
            has_triangulated,
            bfs_scaled,
            best_conf,
            visit_scaled,
            degree_scaled,
        ], dtype=np.float32)