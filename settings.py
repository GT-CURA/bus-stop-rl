class S: 
    """ Streetview Properties """
    key_path = "keys.txt"               # Change to your API key location
    run_server = False                  # Launch webserver, allowing you to view the program in action
    request_msgs = False                # Send messages on start and completion of requests
    max_retries = 5                     # Max number of retries for a failed request
    wait_time = 0                       # Wait time in seconds between steps
    img_height = 640                    # Height of images requested from streetview
    img_width = 640                     # Width of images requested from streetview
    
    """ Stop Loader Properties """
    shuffle_stops = False               # Randomly shuffle stops
    scramble_stops = False              # Randomly move around if stop is visible upon loading
    before_scrambling =-1               # How many stops to load before starting to scramble stops
    min_score_to_scramble = 0.5         # If best evidence of a stop exceeds this, scramble the stop
    loop_stops = False                  # Loop back to the beginning if we run out of stops

    """ RL Properties """
    img_size = (640,640)                # Size that images are compressed to before plugged into YOLO 
    max_steps = 80                      # Max number of steps before forcibly moved to next stop
    min_steps = 35                      # How many steps the model must take before giving up on a stop            
    dampen_scalor = .6                  # How much each score is dampened by
    free_spacebar_presses = 2           # How many times the model can return to start (press spacebar) before being punished
    free_steps_after_found = 5          # Start  punishing model after this many steps since finding stop
    max_steps_after_found = 10          # The number of steps after "found" that the model is allowed before forcibly moving on
    stack_sz = 40                       # Number of frames stacked
    min_conf = .75                      # The minimum confidence value required to be considered "found"

    """ Incentives / Penalties """
    after_found_punishment = .15        # How much to punish model per step after ^
    move_on_reward = .1                 # Points model gets for successfully moving to next episode
    efficiency_bonus = .4               # Additional points for moving on before using all free steps    
    size_scalar = 7                     # The scalar by which change in box size is multiplied and added to score
    max_sz_pts = .2                     # The most amount of additional points from increasing box size
    reused_vp_penalty = .15             # Penalty per viewpoint reuse 
    premature_end = -.7                 # The 'punishment' score model receives for ending early
    consecutive_boost = .2              # How much the model is rewarded for consecutive observations of a stop
    spacebar_penalty = .3               # Model is punished this much per spacebar press after allowed number of presseses
    found_boost = .25                   # Bonus for finding stop

    """ PPO Properties """
    bbs_kept = 3                        # How many of the highest conf bounding boxes will be kept per frame
    action_map = {
        0: "w",
        1: "a",
        2: "s",
        3: "d",
        4: "=",
        5: "Key.enter",
        6: "Key.space"
    }

    """ Logging """
    save_best_img = True                # Save imgs of "best evidence" of each bus stop?
    annotate_best_img = False           # Run YOLO model to annotate saved imgs?
    save_folder = "runs"                # Path to save logs and imgs into  


    """ YOLO Properties"""
    num_classes = 4                     # Number of classes in YOLO model
    yolo_path = "assets/YOLO.pt"        # Path to YOLO model 
    secondary_boost = .35               # How much of the secondary amenities' scores are kept 

    """ API Settings """
    rotate_amt = 45                     # Amount camera angle is changed by on horizontal movement
    dist = 10                           # Distance in meters to search for next pano at when moving forwards/backwards

    """ Don't Touch """
    bb_dim = 4                          # Vector containing bounding box cords, area, class
    features_dim = 256                  # Vector containing YOLO features
    geo_dim = 8                         # Vector containing lat/lon
    frame_dim = features_dim + bbs_kept * (bb_dim + num_classes) + geo_dim
    from datetime import datetime
    log_dir = f"{save_folder}/{datetime.now().strftime('%m-%d_%H-%M-%S')}/"