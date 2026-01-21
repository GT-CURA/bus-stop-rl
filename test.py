import keyboard
import time
from src.streetview.sv import StreetView, Stop
from src.utils.server.server import start_server
from src.rl_env.episode import Episode
from src.stop_detector import StopDetector

# Cobb PKWY: 33.903458, -84.487905
# Downtown: 33.757545, -84.387770
# West Midtown: 33.782396, -84.407611
# Neighborhood /w False pos: 33.770270, -84.413579
# Weird glitch: 33.739878, -84.35477
# Semi-seat 33.732169, -84.306032
# Virginia Highlands 33.796298, -84.350309
# Spawn in gas station: 33.91372, -84.37955
# Triggers point snapping: 33.936201,-84.337755
# Stuck: (33.758338, -84.347739)
# Shelter: 33.726259, -84.392056

DEFAULT_LAT = 33.796298
DEFAULT_LNG = -84.350309
START_HEADING = 90

# === Image Navigator Thread ===
vps = []
def get_vp(sv):
    # Get spatial info from SV URL
    pic = sv.current_pic
    lat, lng, heading = pic.lat, pic.lng, pic.heading
    vp = (round(lat, 6), round(lng, 6), round(heading) % 360)
    vps.append(vp)

def streetview_control():
    sv = StreetView()
    stop_detector = StopDetector(sv)
    stop = Stop(DEFAULT_LAT, DEFAULT_LNG, None, None, None, None)
    sv.goto_pt(stop)
    sv.set_start()
    img = sv.get_img()

    # Simulate getting initial features
    ep = Episode(stop, stop_detector, sv.current_pic)
    output = stop_detector.run(img)
    ep.get_features(img, output, sv.current_pic)
    stop_detector.score_output(
        output, 
        ep.current_node, 
        sv.current_pic, 
        0, 
        False)
    
    print("\n[Street View Controls Ready]")

    while True:
        try:
            action = None
            if keyboard.is_pressed('w'):
                action = 'Forwards'
            elif keyboard.is_pressed('s'):
                action = 'Backwards'
            elif keyboard.is_pressed('a'):
                action = 'Counterclockwise'
            elif keyboard.is_pressed('d'):
                action = 'Clockwise'
            elif keyboard.is_pressed('left'):
                action = "Left"
            elif keyboard.is_pressed('right'):
                action = "Right"
            elif keyboard.is_pressed('q'):
                print("Exiting control thread.")
                break
            elif keyboard.is_pressed("="):
                action = 'Zoom'
            elif keyboard.is_pressed("space"):
                action = "space"
                sv.goto_start()

            if action:
                print(f"Doing action: {action}")
                if action != "space":
                    sv.do_action(action)
                img = sv.get_img()

                # Update episode
                ep.update(action, img, sv.current_pic)
                get_vp(sv)

            time.sleep(0.05)

        except KeyboardInterrupt:
            break

# === Launch Threads ===
if __name__ == "__main__":
    start_server()

    # Run Street View controller in main thread
    streetview_control()