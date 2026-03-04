import os
import json
import threading
import time
import csv
from dataclasses import asdict, fields
from settings import S

AMENITIES = ["sign", "shelter", "trash can", "seating"]

class LogManager:
    def __init__(self, flush_every=5, flush_interval=100):
        self.flush_every = flush_every
        self.flush_interval = flush_interval
        self.buffer = []
        self.visited_buffer = []
        self.lock = threading.Lock()
        self.shutdown_flag = False

        os.makedirs(S.log_dir, exist_ok=True)
        self.path = os.path.join(S.log_dir, "log.csv")
        self.visited_path = os.path.join(S.log_dir, "visited.json")

        # Write header if file does not exist
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames())
                writer.writeheader()

        # Initialize visited.json as empty list if needed
        if getattr(S, "visited_log", False) and not os.path.exists(self.visited_path):
            with open(self.visited_path, "w", encoding="utf-8") as f:
                json.dump([], f)

        # Background flush thread
        self.flush_thread = threading.Thread(
            target=self._background_flush,
            daemon=True
        )
        self.flush_thread.start()

    def add(self, episode):
        """
        Called once per completed episode.
        """
        record = self._build_record(episode)
        visited_record = self._build_visited_record(episode) if getattr(S, "visited_log", False) else None

        should_flush = False
        with self.lock:
            self.buffer.append(record)
            if visited_record is not None:
                self.visited_buffer.append(visited_record)
            if len(self.buffer) >= self.flush_every:
                to_write = self.buffer[:]
                visited_to_write = self.visited_buffer[:]
                self.buffer.clear()
                self.visited_buffer.clear()
                should_flush = True

        if should_flush:
            self._flush_to_disk(to_write)
            if getattr(S, "visited_log", False):
                self._flush_visited_to_disk(visited_to_write)

    def shutdown(self):
        """Flush everything and stop background thread."""
        self.shutdown_flag = True
        self.flush_thread.join()
        self._flush_to_disk()
        if getattr(S, "visited_log", False):
            self._flush_visited_to_disk()

    def _build_visited_record(self, episode):
        """
        Builds a record of all visited nodes and hypotheses. 
        """
        graph = episode.graph

        # Disabled. Uncomment in serialize_node if you want detections. 
        def serialize_det(det):
            d = {k: getattr(det, k, None) for k in [
                "label", "primary_conf", "bearing",
                "timestamp", "pano_id", "lat", "lng",
                "side", "date"
            ]}
            # Ensure all values are JSON serializable
            return {k: (float(v) if isinstance(v, (float, int)) and not isinstance(v, bool) else v)
                    for k, v in d.items()}

        def serialize_node(node):
            return {
                "pano_id": getattr(node, "pano_id", None),
                "lat": node.lat,
                "lng": node.lng,
                "visits": node.visits,
                "best_conf": node.best_conf,
                "best_bearing": node.best_bearing,
                "scores": node.scores,
                # "detections": [serialize_det(d) for d in node.detections],
            }

        def serialize_hyp(hyp):
            return {
                "score": hyp.score,
                "label": hyp.label,
                "side": hyp.side,
                "last_seen": hyp.last_seen,
                "best_bearing": hyp.best_bearing,
                "triangulated_pos": list(hyp.triangulated_pos) if hyp.triangulated_pos else None,
            }

        return {
            "name": episode.stop.place_name,
            "latitude": episode.stop.og_lat,
            "longitude": episode.stop.og_lng,
            "nodes": {pano_id: serialize_node(node) for pano_id, node in graph.graph.items()},
            "hypotheses": [serialize_hyp(h) for h in graph.hypotheses],
        }

    def _build_record(self, episode):
        graph = episode.graph

        # Default empty record
        record = {
            "name": episode.stop.place_name,
            "latitude": episode.stop.og_lat,
            "longitude": episode.stop.og_lng,
            "steps": episode.steps,
            "found_step": None,
            "est_lat": 0,
            "est_lng": 0
        }

        # Initialize amenity scores
        for a in AMENITIES:
            record[a] = 0.0

        # Set date to spawn node initially
        record["date"] = episode.spawn_date

        # No hypotheses, log highest scoring detections
        if not graph.hypotheses:
            best_det = {}
            for node in graph.graph.values():
                for det in node.detections:
                    if det.label not in best_det or det.primary_conf > best_det[det.label].primary_conf:
                        best_det[det.label] = det

            best_date = None
            best_conf_overall = -1.0
            for label, det in best_det.items():
                record[label] = round(float(det.primary_conf), 3)
                if det.primary_conf > best_conf_overall:
                    best_conf_overall = det.primary_conf
                    best_date = getattr(det, "date", None)

            if best_date:
                record["date"] = best_date

            return record

        # Best hypothesis = stop identity
        best_hyp = max(graph.hypotheses, key=lambda h: h.score)

        stop_panos = {d.pano_id for d in best_hyp.observations}
        expanded_panos = set(stop_panos)

        # Include immediate neighbors
        for pano_id in stop_panos:
            node = graph.graph.get(pano_id)
            if node:
                expanded_panos |= node.neighbors

        # Track best detection per amenity
        best_det = {}

        for node in graph.graph.values():
            for det in node.detections:

                # Ensure same side of street
                if det.side != best_hyp.side:
                    continue

                # Ensure its one of the panos we're checking
                if det.pano_id not in expanded_panos:
                    continue

                # Keep highest confidence
                if (
                    det.label not in best_det or
                    det.primary_conf > best_det[det.label].primary_conf
                ):
                    best_det[det.label] = det

        # Populate scores + date
        best_date = None
        best_conf_overall = -1.0

        for label, det in best_det.items():
            record[label] = round(float(det.primary_conf), 3)

            # Choose date from strongest overall evidence
            if det.primary_conf > best_conf_overall:
                best_conf_overall = det.primary_conf
                best_date = getattr(det, "date", None)

        # Record date from best hyp 
        record["date"] = best_date
        
        # Total number of steps - steps since found + 1 step for spawn + 1 step for next 
        record["found_step"] = episode.steps - episode.steps_since_found + 2

        # Record estimated coords
        if best_hyp.triangulated_pos:
            hyp_x, hyp_y = best_hyp.triangulated_pos
            est_lng, est_lat = episode.context.to_global.transform(hyp_x, hyp_y)
            record["est_lat"] = est_lat
            record["est_lng"] = est_lng
        return record

    def _flush_to_disk(self, rows=None):
        if rows is None:
            with self.lock:
                rows = self.buffer[:]
                self.buffer.clear()

        if not rows:
            return

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames())
            for row in rows:
                writer.writerow(row)

    def _flush_visited_to_disk(self, rows=None):
        if rows is None:
            with self.lock:
                rows = self.visited_buffer[:]
                self.visited_buffer.clear()

        if not rows:
            return

        # Read existing records, append new ones, write back
        existing = []
        if os.path.exists(self.visited_path):
            try:
                with open(self.visited_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []

        existing.extend(rows)

        with open(self.visited_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)

    def _background_flush(self):
        while not self.shutdown_flag:
            time.sleep(self.flush_interval)
            self._flush_to_disk()
            if getattr(S, "visited_log", False):
                self._flush_visited_to_disk()

    @staticmethod
    def _fieldnames():
        return ["name", "latitude", "longitude", "date", "steps", "found_step", "est_lat", "est_lng"] + AMENITIES