class S: 
    """ Streetview Properties """
    key_path = "keys.txt"               # Change to your API key location
    run_server = True                   # Launch webserver, allowing you to view the program in action=
    max_retries = 5                     # Max number of retries for a failed request
    wait_time = 0                       # Wait time in seconds between steps
    img_height = 640                    # Height of images requested from streetview
    img_width = 640                     # Width of images requested from streetview
    cache_dir = "graph_cache"           # Directory to save OSMNX graph cache in 
    pic_caching = False                 # Save images for later reuse
    
    """ Stop Loader Properties """
    shuffle_stops = True                # Randomly shuffle stops
    scramble_stops = True               # Randomly move around if stop is visible upon loading
    before_scrambling = -1              # How many stops to load before starting to scramble stops
    min_score_to_scramble = 0.5         # If best evidence of a stop exceeds this, scramble the stop
    loop_stops = True                   # Loop back to the beginning if we run out of stops

    """ RL Properties """
    img_size = (640,640)                # Size of images pulled from GSV 
    out_size = 64                       # Size that images are compressed to before plugged into agent
    max_steps = 80                      # Max number of steps before forcibly moved to next stop
    min_steps = 75                      # How many steps the model must take before giving up on a stop            
    dampen_scalor = .6                  # How much each score is dampened by
    free_spacebar_presses = 2           # How many times the model can return to start (press spacebar) before being punished
    free_steps_after_found = 12         # Start  punishing model after this many steps since finding stop
    max_steps_after_found = 25          # The number of steps after "found" that the model is allowed before forcibly moving on
    stack_sz = 80                       # Number of frames stacked
    min_conf = .75                      # The minimum confidence value required to be considered "found"

    """ Incentives / Penalties """
    after_found_punishment = .05        # How much to punish model per step after ^
    move_on_reward = .1                 # Points model gets for successfully moving to next episode
    efficiency_bonus = .4               # Additional points for moving on before using all free steps    
    premature_end = -.3                 # The 'punishment' score model receives for ending before finding a stop
    spacebar_penalty = .1               # Model is punished this much per spacebar press after allowed number of presseses
    found_boost = .2                    # Bonus for finding stop
    heading_weight = .3                 # Weight for rotating towards observations  
    graph_weight = .1                   # Weight for returning to node with best obesrvations
    coord_weight = .05                  # Weight for moving towards estimated Stop coord 
    new_node_bonus = .05                # Points rewarded for visiting a new node (encourage exploration)
    undo_penalty = .02                  # Prevent agent from oscillating its actions 
    linger_penalty = .02                # Punish the model for staying at the same pano for too long (not moving)
    panning_penalty = .02
    zoom_cost = .01                    # Slight penalty to prevent zoom spamming

    """ PPO Properties """
    bbs_kept = 3                        # How many of the highest conf bounding boxes will be kept per frame
    action_map = {
        0: "Forwards",
        1: "Counterclockwise",
        2: "Backwards",
        3: "Clockwise",
        4: "Zoom",
        5: "Return",
        6: "Next"
    }

    """ Logging """
    save_best_img = True                # Save imgs of "best evidence" of each bus stop?
    annotate_best_img = False           # Run YOLO model to annotate saved imgs?
    save_folder = "runs"                # Path to save logs and imgs into  

    """ Console Msgs """
    msg_score_breakdown = False         # Show spatial scoring elements
    request_msgs = False                # Send messages on start and completion of requests
    yolo_msgs = False                   # Show console output from YOLO model

    """ YOLO Properties"""
    num_classes = 4                     # Number of classes in YOLO model
    yolo_path = "assets/YOLO.pt"        # Path to YOLO model 
    secondary_prefound = .25            # How much of the secondary amenities' scores are kept before finding
    primary_found = .5                  # How much of primary amenities' scores are kept after finding

    """ API Settings """
    rotate_amt = 45                     # Amount camera angle is changed by on horizontal movement
    dist = 10                           # Distance in meters to search for next pano at when moving forwards/backwards

    """ Don't Touch """
    bb_dim = 4                          # Vector containing bounding box cords, area, class
    bb_total_dim = bbs_kept * (bb_dim + num_classes)
    features_dim = 256                  # Vector containing YOLO features

    geo_dim_basic = 6                   # Basic spatial info like lat/lng 
    geo_dim_graph = 8                   # Spatial feature vector (from graph class)
    geo_dim = geo_dim_basic + geo_dim_graph
    img_dim = out_size * out_size
    frame_dim = features_dim + bb_total_dim + geo_dim + img_dim

    from datetime import datetime
    log_dir = f"{save_folder}/{datetime.now().strftime('%m-%d_%H-%M-%S')}/"