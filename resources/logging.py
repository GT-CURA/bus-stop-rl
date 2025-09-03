import os
import threading 
import json
import time
from settings import S

class LogManager:
    def __init__(self, flush_every=5, flush_interval=100):
        self.flush_every = flush_every 
        self.flush_interval = flush_interval
        self.buffer = []
        self.lock = threading.Lock()
        self.shutdown_flag = False
        self.path = f"{S.log_dir}log.json"

        # Create directory
        os.makedirs(S.log_dir, exist_ok=True)
        
        # Background thread to flush periodically
        self.flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self.flush_thread.start()

    def add(self, instance):
        """Add a single episode record to the log buffer."""
        record = {
            "place_name": instance.stop.place_name,
            "latitude": instance.stop.og_lat,
            "longitude": instance.stop.og_lng,
            "amenity_scores": instance.amenity_scores,
            "total_reward": round(instance.reward, 3),
            "steps_taken": instance.steps
        }

        with self.lock:
            self.buffer.append(record)
            if len(self.buffer) >= self.flush_every:
                self._flush_to_disk()

    def _flush_to_disk(self):
        if not self.buffer:
            return

        with self.lock:
            to_write = self.buffer[:]
            self.buffer.clear()

        if os.path.exists(S.log_dir):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                data = []
        else:
            data = []

        data.extend(to_write)

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _background_flush(self):
        while not self.shutdown_flag:
            time.sleep(self.flush_interval)
            self._flush_to_disk()

    def shutdown(self):
        """Gracefully shut down and write any remaining logs."""
        self.shutdown_flag = True
        self.flush_thread.join()
        self._flush_to_disk()