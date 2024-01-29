#%%
import pandas as pd
import os
import numpy as np

data_root = ""
cur_case = 'user2_data_radar' # radar experiment data folder
target_data_path = "0128Aligned" # folder to save aligned data

def find_alignment_indices(timestamps1, timestamps2):
    """
    Find the starting alignment indices for two lists of timestamps.
    
    :param timestamps1: First list of timestamps.
    :param timestamps2: Second list of timestamps.
    :return: A tuple containing the index in the first list and the index in the second list where the alignment starts.
    """
    min_diff = float('inf')
    align_index_1 = -1
    align_index_2 = -1
    
    for i, t1 in enumerate(timestamps1):
        # Find the closest timestamp in the second list and its index
        closest_diff, j = min((abs(t1 - t2), idx) for idx, t2 in enumerate(timestamps2))
        
        # Update the align_index if this is the smallest difference found so far
        if closest_diff < min_diff:
            min_diff = closest_diff
            align_index_1 = i
            align_index_2 = j
            
    return align_index_1, align_index_2

def create_folder(path, folder_name):
    """Create a new folder under a specified path."""
    folder_path = os.path.join(path, folder_name)
    
    # Check if directory already exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        print(f"Directory {folder_name} already exists at {path}")

    return folder_path

# Change here to select correct Neulog data
user_id = "1"
distance = [0.8]
degree = [0]

for i in distance:
    for j in degree:
    
        cur_scenario = 'user{}_distance_{}m_{}'.format(user_id, i, j) # radar data folder name
        data_path = os.path.join(data_root, cur_case, cur_scenario)
        recording = np.load(os.path.join(data_path, "recording.npy"))
        config = np.load(os.path.join(data_path, "config.npy"), allow_pickle=True).item()
        calibration = np.load(os.path.join(data_path, "calibration.npy"))

        start_time = config['collect_time']
        sample_time = config['sample_time']
        len_frame = config['len_frame']

        list_of_ts = [start_time + sample_time for i in range(len_frame+1)]
        timestamp_list1 = list_of_ts
        neulog_df = pd.read_csv("{}_front_{}_{}.csv".format(user_id, i, j), header=None)
        timestamp_list2 = list(neulog_df[0].values)

        align_index_1, align_index_2 = find_alignment_indices(timestamp_list1, timestamp_list2)
        print(align_index_1, align_index_2)

        print(timestamp_list1[align_index_1], timestamp_list2[align_index_2])

        folder_name = cur_scenario + "_" + cur_case
        new_folder_path = create_folder(target_data_path, folder_name)

        new_reading = recording[align_index_1:,:,:]
        new_respiration_gt = np.array(list(neulog_df[1].values)[align_index_2:])


        np.save(os.path.join(target_data_path, folder_name, "recording.npy"), new_reading)
        np.save(os.path.join(target_data_path, folder_name, "respiration_gt.npy"), new_respiration_gt)
        np.save(os.path.join(target_data_path, folder_name, "calibration.npy"), calibration)
        np.save(os.path.join(target_data_path, folder_name, "config.npy"), config)