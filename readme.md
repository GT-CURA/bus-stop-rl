# Bus Stop Reinforcement Learning
Using reinforcement learning to find bus stops! 

## Setup
1. Install required packages:
 - Stable Baselines (for PPO algorithm)
 - Ultralytics (for YOLO model)
 - Flask (for serving images)

2. Setup Street View API:
 - Get an API key from Google[https://developers.google.com/maps/documentation/streetview/get-api-key?setupProd=enable]
 - Save your key(s) to a txt file 
 - Place the path of the txt file in settings.py under key_path

3. Setup a JSON/CSV file of stops:
 - All stops must have "longitude," "latitude" and "name" attributes. The "name" value is used for logging purposes and can be any value. 

4. (Optional) Configure Flask server to watch the model in action:
 - In settings.py, change run_server to True. 
 - Increase wait_time in settings to add some buffer time between steps. Otherwise, the program might get hung up while trying to save images. 
 - In the 'main' block of run.py, specify a port in the start_server() method call, or leave as 5000
 - After starting the program, navigate to localhost:{your port here} in a browser

## Training
Navigate to the 'main' block in run.py (located at the end of the file). Use the train() method, specifying a path to save the model in, a csv or json file of bus stops, and, if resuming training, an existing model. 

### Stop Loading
To prevent the task of finding stops from being too simple, I wrote logic to "scramble" stops. After pulling up a new stop, the YOLO model is run to asses if a sign or shelter is within sight. If a shelter or sign is detected with a confidence value of at least S.min_score_to_scramble (specified in settings.py), the program will move in a random direction and pick a new camera angle. This process repeats recursively up to three times until the stop is no longer in view. 
