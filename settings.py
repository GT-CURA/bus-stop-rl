class S: 
    """ Streetview Properties """
    key_path = "keys.txt"               # Change to your API key location
    show_imgs = False                   # Whether to display images on web server
    request_msgs = False                # Send messages on start and completion of requests
    max_retries = 5                     # Max number of retries for a failed request
    img_height = 640
    img_width = 640
    
    """ Stop Loader Properties """
    shuffle_stops = True                
    scramble_positive_stops = True      # Randomly move around after loading a positive stop
    before_scrambling = 50              # How many stops to load before starting to scramble positive stops
    num_positives = 8000                # How many positive stops to include in training
    min_score_to_scramble = 0.5         # If best evidence of a stop exceeds this, scramble the stop

    """ YOLO Properties"""
    num_classes = 4
    yolo_path = "assets/YOLO.pt"
    secondary_boost = .35               # How much of the secondary amenities' scores are kept 

    """ RL Properties """
    img_size = (640,640)                # Size that images are compressed to before plugged into YOLO 
    max_steps = 40                      # Max number of steps before forcibly moved to next stop
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

    """ Logging & Screenshots """
    save_screenshots = False           # Save screenshots of "best evidence" of each bus stop?
    annotate_screenshots = False       # Run YOLO model to annotate screenshots?
    save_folder = "runs"


    """ API Settings """
    rotate_amt = 45

    """ Don't Touch """
    bb_dim = 4                          # Vector containing bounding box cords, area, class
    features_dim = 256                  # Vector containing YOLO features
    geo_dim = 8                         # Vector containing lat/lon
    frame_dim = features_dim + bbs_kept * (bb_dim + num_classes) + geo_dim
    from datetime import datetime
    log_dir = f"{save_folder}/{datetime.now().strftime('%m-%d_%H-%M-%S')}/"