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

3. (Optional) Configure Flask server to watch the model in action:
 - In settings.py, change run_server to True. 
 - Increase wait_time in settings to add some buffer time between steps. Otherwise, the program might get hung up while trying to save images. 
 - In the 'main' block of run.py, specify a port in the start_server() method call, or leave as 5000
 - After starting the program, navigate to localhost:{your port here} in a browser

## Training
Navigate to the 'main' block in run.py (located at the end of the file). Use the train() method, specifying   