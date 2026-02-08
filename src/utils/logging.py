import os
import threading
import time
import csv
from settings import S
from src.utils.context import RoadContext

AMENITIES = ["sign", "shelter", "trash can", "seating"]

class LogManager:
    def __init__(self, context = RoadContext, flush_every=5, flush_interval=100):
        self.context = context
        self.flush_every = flush_every
        self.flush_interval = flush_interval
        self.buffer = []
        self.lock = threading.Lock()
        self.shutdown_flag = False

        os.makedirs(S.log_dir, exist_ok=True)
        self.path = os.path.join(S.log_dir, "log.csv")

        # Write header if file does not exist
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames())
                writer.writeheader()

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

        should_flush = False
        with self.lock:
            self.buffer.append(record)
            if len(self.buffer) >= self.flush_every:
                to_write = self.buffer[:]
                self.buffer.clear()
                should_flush = True

        if should_flush:
            self._flush_to_disk(to_write)

    def shutdown(self):
        """Flush everything and stop background thread."""
        self.shutdown_flag = True
        self.flush_thread.join()
        self._flush_to_disk()

    def _build_record(self, episode):
        graph = episode.graph

        # Default empty record
        record = {
            "name": episode.stop.place_name,
            "latitude": episode.stop.og_lat,
            "longitude": episode.stop.og_lng,
            "date": None,
            "steps": episode.steps,
            "est_lat": 0,
            "est_lng": 0
        }

        # Initialize amenity scores
        for a in AMENITIES:
            record[a] = 0.0

        # No hypotheses → nothing to log
        if not graph.hypotheses:
            return record

        # Best hypothesis = stop identity
        best_hyp = max(graph.hypotheses, key=lambda h: h.score)

        stop_panos = {d.pano_id for d in best_hyp.observations}
        expanded_panos = set(stop_panos)

        # Include immediate neighbors (optional but recommended)
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

        # Record estimated coords
        if best_hyp.triangulated_pos:
            hyp_x, hyp_y = best_hyp.triangulated_pos
            est_lat, est_lng = self.context.to_global.transform(hyp_x, hyp_y)
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

    def _background_flush(self):
        while not self.shutdown_flag:
            time.sleep(self.flush_interval)
            self._flush_to_disk()

    @staticmethod
    def _fieldnames():
        return ["name", "latitude", "longitude", "date", "steps", "est_lat", "est_lng"] + AMENITIES
