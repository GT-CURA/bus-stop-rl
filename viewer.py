import threading
import keyboard
import cv2
import time
from flask import Flask, render_template
from pathlib import Path
from resources.streetview import StreetView, Stop
import logging

# === Flask setup ===
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True
@app.route('/')
def index():
    return render_template('index.html')

IMG_PATH = Path("static/frame.jpg")
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
                img = sv.get_img()
                IMG_PATH.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(IMG_PATH), img)
                time.sleep(0.25)

            time.sleep(0.05)

        except KeyboardInterrupt:
            break

# === Launch Threads ===
if __name__ == "__main__":
    # Start Flask app in background
    flask_thread = threading.Thread(target=lambda: app.run(debug=False, use_reloader=False))
    flask_thread.start()

    # Run Street View controller in main thread
    streetview_control()
