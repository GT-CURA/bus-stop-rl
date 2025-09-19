class S: 
    """ Streetview Properties """
    show_imgs = True                    # Whether to display images
    wait_time = 2                       # How long to wait between images
    sleep_time = 1                      # Sleep between requests
    request_msgs = False

    """ YOLO Properties"""
    num_classes = 5
    yolo_path = "assets/YOLO.pt"
    secondary_boost = .35               # How much of the secondary amenities' scores are kept 

    """ RL Properties """
    img_size = (640,640)                # Size that images are compressed to before plugged into YOLO 
    max_steps = 35                      # Max number of steps before forcibly moved to next stop
    min_steps = 25                      # How many steps the model must take before giving up on a stop            
    batch_size = 10
    dampen_scalor = .6                  # How much each score is dampened by
    premature_end = -.7                 # The 'punishment' score model receives for ending early
    consecutive_boost = .2              # How much the model is rewarded for multiple pics of the stop
    free_spacebar_presses = 2           # How many times the model can return to start (press spacebar) before being punished
    spacebar_penalty = .3               # Model is punished per spacebar press after allowed number of presseses
    free_steps_after_found = 7          # Start heavily punishing model after this many steps since finding stop
    max_steps_after_found = 20          # The number of steps after "found" that the model is allowed before forcibly moving on
    after_found_punishment = .15        # How much to punish model per step after ^
    move_on_reward = .1                 # Points model gets for successfully moving to next episode
    efficiency_bonus = .3               # Additional points for moving on before using all free steps    
    size_scalar = 7                     # The scalar by which change in box size is multiplied and added to score.
    max_sz_pts = .2                     # The most amount of additional points from increasing box size
    multi_persp_reward = .3             # Points for having found multiple perpsectives of the stop
    num_persp_rewarded = 4              # Max number of perspectives the model is rewarded for finding
    stack_sz = 30

    """ RPPO Properties """
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
    save_screenshots = True            # Save screenshots of "best evidence" of each bus stop?
    annotate_screenshots = False       # Run YOLO model to annotate screenshots?
    save_folder = "runs"


    """ API Settings """
    rotate_amt = 45

    """ Don't Touch """
    bb_dim = 4                          # Vector containing bounding box cords, area, class
    features_dim = 512                  # Vector containing YOLO features
    geo_dim = 9                         # Vector containing lat/lon
    frame_dim = features_dim + bbs_kept * (bb_dim + num_classes) + geo_dim
    from datetime import datetime
    log_dir = f"{save_folder}/{datetime.now().strftime('%m-%d_%H-%M-%S')}/"