from settings import S 
import json
from random import sample, shuffle, randint
from resources.streetview import StreetView
import time
from resources.stop import Stop

class StopLoader:

    def __init__(self, streetview):
        self.sv = streetview
        self.index = 0
        self.stops = None
        
    def load_stops(self, path: str, shuffle_stops = True, num_positives=0, ignore_path: str = None):
        # Load all stops 
        with open(path) as f:
                scores = json.load(f)
        
        # Find which stops to ignore if specified
        if ignore_path: 
            with open(ignore_path) as f:
                ignore_f = json.load(f)
            
            stops_ignored = []
            for score_num in ignore_f:
                stops_ignored.append(score_num["place_name"])

        # Go through stops
        stops = []
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
        time.sleep(1 + S.standard_wait)

        # If we couldn't load the stop, re-run function
        if not loaded: 
            self.load_stop()
            return stop

        # Make streetview accept input
        if wiggle_mouse:
            self.sv.wiggle_mouse()

        # If stop is a positive, scramble
        if not stop.false_negative:
            self.scramble_positive()

        # Tell SV to log initial position
        self.sv.set_start()
        return stop

    """ WIP way to use positive stops to train """
    def scramble_positive(self):
        print("\n[Stop Loader] Scrambling positive stop...")

        # Pick a direction to walk in, press key x times
        action = sample(['w','s'], 1)
        self.press_loop(action, randint(0, 5))

        # Pick a direction to turn in, press key x times
        action = sample(['a','d'], 1)
        self.press_loop(action, randint(0,3))
        print("[Stop Loader] Complete!\n")

    def press_loop(self, action: str, num: int):
        for i in range(num):
            self.sv.do_action(action[0])