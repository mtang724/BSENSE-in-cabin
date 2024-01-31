# Step to Collect Radar data

### Download Vayyar Vtrig Driver:
1. Go to: 
https://www.minicircuits.com/WebStore/imagevk_software_down
2. Input serial number located at the back of radar
3. Input some personal information, then download the driver

### Install the driver
1. Click the .exe file, excute installation process
2. When installation done, go to C:\Program Files\Vayyar\vtrigU\python\vtrigU-1.4.3.zip\vtrigU-1.4.3
3. Install Python lib by `pip install .` inside vtrigU-1.4.3 folder

### Run collecting_data
Config ground truth: 
```
gt = 0 # respiration ground truth
degree = 0 # orientation
round = 1 # num of trails
calibrate = False # True for calibration, False for Recording
len_frame = 300 # 30s - 40s # Number of frames
situation = "" # rear_w_reflector
case = "radar_data" # Radar data folder name
distance = 0.5 # in m
```
# Step to NeuLog Data Collection


### Download NeuLog API
https://neulog.com/Downloads/neulog_api_ver_004.exe

### Run collecting NeuLog Data
**Note: When you collecting the actual data, please run this one and radar collection simultaneously**

open NeuLog API, conform the the status is **Ready**
`Python NeuLogRespirationLogger.py`

Parameters that can change: 
```
user_id = "1"
distance1 = 0.8 # in meters
degree = 0 # in degrees
```

### Align NeuLog Data with Radar data
`Python align_vtrigu_neulog.py`

#### For example, radar data in  'radar_data/{}k_profile_{}_distance{}_degree_{}_round_{}', NeuLog file name '{}_front_{}_{}.csv'.
Edit the following parameters.
```
cur_case = 'radar_data' # radar experiment data folder
target_data_path = "0128Aligned" # folder to save aligned data
i for distance
j for degree
cur_scenario = '{}k_profile_{}_distance_degree_{}_round_{}'.format(50, 1, i, j, 1) # radar data folder name
neulog_df = pd.read_csv("{}_front_{}_{}.csv".format(round, i, j), header=None) # ground truth
```