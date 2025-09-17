import keyboard
import time
from resources.streetview import StreetView, Stop
from resources.server import start_server

DEFAULT_LAT = 33.7738236  
DEFAULT_LNG = -84.3816805 
START_HEADING = 90

# === Image Navigator Thread ===
def streetview_control():
    sv = StreetView()
    sv.launch("key.txt")

    stop = Stop(DEFAULT_LAT, DEFAULT_LNG, None, None, None, None)
    sv.goto_pt(stop)
    sv.set_start()

    print("\n[Street View Controls Ready]")

    while True:
        try:
            action = None
            if keyboard.is_pressed('w'):
                action = 'w'
            elif keyboard.is_pressed('s'):
                action = 's'
            elif keyboard.is_pressed('a'):
                action = 'a'
            elif keyboard.is_pressed('d'):
                action = 'd'
            elif keyboard.is_pressed('q'):
                print("Exiting control thread.")
                break
            elif keyboard.is_pressed("="):
                action = '='
            if action:
                print(f"Doing action: {action}")
                sv.do_action(action)
                sv.get_img()
                time.sleep(0.25)

            time.sleep(0.05)

        except KeyboardInterrupt:
            break

# === Launch Threads ===
if __name__ == "__main__":
    start_server()

    # Run Street View controller in main thread
    streetview_control()
