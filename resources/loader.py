from settings import S 
import json
from random import sample, shuffle, randint
from resources.stop import Stop
from resources.stop_detector import StopDetector
import csv
class StopLoader:

    def __init__(self, streetview, scramble_pos=False):
        self.sv = streetview
        self.index = 0
        self.stops = None
        self.scramble_pos = scramble_pos
        self.stop_detector: StopDetector = None
        
    def load_stops(self, path: str, shuffle_stops = True, num_positives=0, ignore_path: str = None):
        # Find which stops to ignore if specified
        if ignore_path: 
            with open(ignore_path) as f:
                ignore_f = json.load(f)
            
            stops_ignored = []
            for score_num in ignore_f:
                stops_ignored.append(score_num["place_name"])

        stops = []
        # Load stops (CSV)
        if path.lower().endswith('.csv'):
            with open(path, mode='r', newline='', encoding='utf-8') as csvfile:
                
                # Iterate through stops, creating Stop objects
                reader = csv.DictReader(csvfile)
                for row in reader:
                    stop = Stop(float(row["latitude"]), float(row["longitude"]),
                                row["name"], None, False, None)

                    # Ignore if requestted
                    if ignore_path:
                        if score["name"] in stops_ignored:
                            continue
                    stops.append(stop)

        else:             
            # Load all stops 
            with open(path) as f:
                    scores = json.load(f)

            # Go through stops
            pos = []
            for score_num in scores:

                # Build stop 
                score = scores[score_num]
                stop = Stop(score["latitude"], score["longitude"], 
                            score["gmaps_place_name"], None, True, None)

                # Check if this is in the ignore list 
                if ignore_path:
                    if score["gmaps_place_name"] in stops_ignored:
                        continue 

                # Pull out false negatives
                if len(score['amenity_scores']) == 0:
                    stops.append(stop)
                else:
                    stop.false_negative = False
                    pos.append(stop)

            # Include positives if requested 
            if num_positives:
                stops.extend(sample(pos, num_positives))
        
        # Shuffle if requested
        if shuffle_stops:
            shuffle(stops)
        self.stops = stops

    def load_stop(self, stop: Stop = None, wiggle_mouse=True):
        # Automatically pull next stop
        if not stop: 
            stop = self.stops[self.index]
            self.index += 1 

        # Build point, try to navigate to it
        loaded = self.sv.goto_pt(stop)

        # If we couldn't load the stop, re-run function
        if not loaded: 
            self.load_stop()
            return stop

        # If stop is a positive, scramble
        if not stop.false_negative and self.scramble_pos:
            self.scramble_positive()

        # Tell SV to log initial position
        self.sv.set_start()
        return stop

    """ WIP way to use positive stops to train """
    def scramble_positive(self, tries = 0):
        print("\n[Stop Loader] Scrambling positive stop...")

        # Pick a direction to walk in, press key x times
        action = sample(['w','s'], 1)
        self.press_loop(action, randint(0, 5))

        # Pick a direction to turn in, press key x times
        action = sample(['a','d'], 1)
        self.press_loop(action, randint(0,3))

        # Check if stop is still visible
        img = self.sv.get_img()
        output = self.stop_detector.run(img)
        _, found, _, _ = self.stop_detector.score_output(output)

        # Stop still visible
        if found:
            # Run function again if tries haven't been exhausted
            if tries < 2:
                self.scramble_positive(tries = tries + 1)
            
            # Turn away from the stop
            else:
                action = sample(['a','d'], 1)
                self.press_loop(action, randint(0,2))
        print("[Stop Loader] Complete!\n")

    def press_loop(self, action: str, num: int):
        for i in range(num):
            self.sv.do_action(action[0])