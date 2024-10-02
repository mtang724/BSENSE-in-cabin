import json
import os
import numpy as np
from utils import sliding_window
from vayyar_preprocess import MVDR_beamforming, range_doppler_processing, MVDR_beamforming_with_phase
from scipy.constants import c
from sklearn.preprocessing import normalize
import csv
import argparse

def data_gen_one_file(data_path, is_doppler_only=False):
    """
    Processes radar data from a single file and generates various outputs including heatmaps, beamformed signals, 
    and range-Doppler maps. The processed data is saved to a specified directory.
    Parameters:
    data_path (str): Path to the directory containing the radar data and metadata.
    is_doppler_only (bool): If True, only Doppler processing is performed. Default is False.
    Returns:
    tuple: A tuple containing a boolean indicating success or failure, and a message string.
    The function performs the following steps:
    1. Reads metadata from a JSON file.
    2. Checks for benchmark and in-car data conditions.
    3. Sets distance parameters based on the dataset location.
    4. Loads radar data and configuration.
    5. Applies Hanning window and FFT to the radar data.
    6. Performs beamforming if `is_doppler_only` is False.
    7. Calculates distance vectors and identifies seat row indexes.
    8. Generates heatmaps and beamformed signals for each seat row.
    9. Performs range-Doppler processing.
    10. Saves the processed data to the specified directory.
    11. Copies the respiration belt CSV file if aligned.
    Example:
    >>> success, message = data_gen_one_file("/path/to/data", is_doppler_only=True)
    >>> print(success, message)
    True, "Successful"
    """
    ## Read Metadata Json file    
    print(data_path)
    file_path = os.path.join(data_path, "metadata.json")  # Update this to the path of your JSON file
    if not os.path.exists(file_path):
        return False, "Not Exist Metadata"
    # Read the JSON file
    with open(file_path, 'r') as file:
        json_data = json.load(file)
        
    if json_data['is_benchmark']:
        return False, "Benchmark Data"
    
    if json_data['in_car']:
        return False, "In Car Data"
    
    casename = "_".join(json_data['exp_comment'].split('/'))


    ## Process one file data
    # Read data
    range_nfft = 128
    radar_window_size = 150
    radar_step_size = 30
    num_tx = 10
    angle_nfft = [4, 4]
    
    #### The row distance is different for each dataset, measured by beamforming method, developed in /Vayyar_Realtime/real_time_visualizer_new.py
    if json_data["is_benchmark"]:
        row1_dist_low = 0.6
        row1_dist_high = 1.0
        row2_dist_low = 1.4
        row2_dist_high = 1.8
        row3_dist_low = 2.3 - 0.3
        row3_dist_high = 2.3 + 0.3
    else:
        if json_data["where"] == "CSL":
            row1_dist_low = 0.6
            row1_dist_high = 1.0
            row2_dist_low = 1.4
            row2_dist_high = 1.8
            row3_dist_low = 2.35
            row3_dist_high = 2.75
        if json_data["where"] == "BMW":
            row1_dist_low = 0.3
            row1_dist_high = 0.7
            row2_dist_low = 1.0
            row2_dist_high = 1.4
            row3_dist_low = 1.9
            row3_dist_high = 2.3
        if json_data["where"] == "Toyota Corolla":
            row1_dist_low = 0.4
            row1_dist_high = 0.8
            row2_dist_low = 1.1
            row2_dist_high = 1.5
            row3_dist_low = 2
            row3_dist_high = 2.4
        if json_data["where"] == "Toyota RAV4":
            row1_dist_low = 0.3
            row1_dist_high = 0.7
            row2_dist_low = 1.0
            row2_dist_high = 1.4
            row3_dist_low = 1.9
            row3_dist_high = 2.3
    azimDim = 16
    elevDim = 16
    row1_beamformed_signals_list = []
    row2_beamformed_signals_list = []
    row3_beamformed_signals_list = []
    row1_heatmap_samples = []
    row2_heatmap_samples = []
    row3_heatmap_samples = []
    rd_maps_list_row1 = []
    rd_maps_list_row2 = []
    rd_maps_list_row3 = []

    recording = np.load(os.path.join(data_path, "recording.npy"))
    config = np.load(os.path.join(data_path, "config.npy"), allow_pickle=True).item()

    processed_data = recording
    # sample_time = config['sample_time']
    hanning_window = np.hanning(processed_data.shape[-1])
    processed_data = processed_data * hanning_window[None, None, :]
    range_profile = np.fft.ifft(processed_data, n=range_nfft, axis=2)
    # num_of_frames = len(range_profile)
    # list_of_frames = np.array(list(range(num_of_frames)))
    processed_data_range_list = sliding_window(range_profile, window_size=radar_window_size, step_size=radar_step_size)
    # processed_frame_list = sliding_window(list_of_frames, window_size=radar_window_size, step_size=radar_step_size)

    num_frames = radar_window_size - 20
    for range_data in processed_data_range_list:
        background = range_data[0:15]
        hanning_window = np.hanning(background.shape[-1])
        background = background * hanning_window[None, None, :]
        cal_range_profile = np.fft.ifft(background, n=range_nfft, axis=2)
        range_data = range_data[20:] - np.mean(cal_range_profile)
        print("range data shape", range_data.shape)
        # FFT Process
        range_data_3d = range_data.reshape(range_data.shape[0], num_tx, 20, range_data.shape[-1])
        processed_data_3d_range = np.fft.fft2(range_data_3d, s=angle_nfft, axes=(1,2))
        
        # Beamforming Process
        if not is_doppler_only:
            range_data_trans = np.transpose(range_data, (1, 0, 2))
            print("range data", range_data_trans.shape)
            rangeAngle, bfWeight, invCovMat, phaseInfo = MVDR_beamforming_with_phase(range_data_trans, num_tx, searchStep=10)
            print("Phase data extracted:", phaseInfo.shape)
        
        ## Dist Parameters
        range_bins = np.arange(range_nfft)  # Your range bins
        freq = config['freq']
        Ts = 1/range_nfft/(freq[1]-freq[0]+1e-16) # Avoid nan checks
        time_vec = np.linspace(0,Ts*(range_nfft-1),num=range_nfft)
        dist_vec = time_vec*(c/2) # distance in meters
        
        ## Seat Row 1 and Row2 indexes
        range_low_row1 = np.where(dist_vec>=row1_dist_low)[0][0]
        range_high_row1 = np.where(dist_vec<=row1_dist_high)[0][-1]
        
        range_low_row2 = np.where(dist_vec>=row2_dist_low)[0][0]
        range_high_row2 = np.where(dist_vec<=row2_dist_high)[0][-1]
        
        range_low_row3 = np.where(dist_vec>=row3_dist_low)[0][0]
        range_high_row3 = np.where(dist_vec<=row3_dist_high)[0][-1]
        
        # Generate pre-process-data
        ## Row1
        ### Heat map
        row1_currHeatmap_list = []
        for rngIdx in range(range_low_row1, range_high_row1):
            tempA = np.squeeze(rangeAngle[rngIdx, :])
            currHeatmap = normalize(tempA.reshape(azimDim, elevDim), axis=1, norm="l1")
            row1_currHeatmap_list.append(currHeatmap)
        row1_heatmap_samples.append(row1_currHeatmap_list)
        ### Beamformed Signal
        row1_beamformed_signals = phaseInfo[:, range_low_row1:range_high_row1, :]
        row1_beamformed_signals_list.append(row1_beamformed_signals) # 150 for 12s
        
        ## Row2
        ### Heat map
        row2_currHeatmap_list = []
        for rngIdx in range(range_low_row2, range_high_row2):
            tempA = np.squeeze(rangeAngle[rngIdx, :])
            currHeatmap = normalize(tempA.reshape(azimDim, elevDim), axis=1, norm="l1")
            row2_currHeatmap_list.append(currHeatmap)
        row2_heatmap_samples.append(row2_currHeatmap_list)
        ### Beamformed Signal
        row2_beamformed_signals = phaseInfo[:, range_low_row2:range_high_row2, :]
        row2_beamformed_signals_list.append(row2_beamformed_signals)
        
        ## Row2
        ### Heat map
        row3_currHeatmap_list = []
        for rngIdx in range(range_low_row3, range_high_row3):
            tempA = np.squeeze(rangeAngle[rngIdx, :])
            currHeatmap = normalize(tempA.reshape(azimDim, elevDim), axis=1, norm="l1")
            row3_currHeatmap_list.append(currHeatmap)
        row3_heatmap_samples.append(row3_currHeatmap_list)
        ### Beamformed Signal
        row3_beamformed_signals = phaseInfo[:, range_low_row3:range_high_row3, :]
        row3_beamformed_signals_list.append(row3_beamformed_signals)
            
        
        rd_maps_row1 = []
        rd_maps_row2 = []
        rd_maps_row3 = []
        # print(processed_data_3d_range.shape[0])
        for i in range(angle_nfft[0]):
            for j in range(angle_nfft[1]):
                rd_map = range_doppler_processing(processed_data_3d_range[:, i, j, :], range_nfft, processed_data_3d_range.shape[0] * 2, config)
                rd_maps_row1.append(rd_map[range_low_row1:range_high_row1, :25]) # 75 for 12s
                rd_maps_row2.append(rd_map[range_low_row2:range_high_row2, :25])
                rd_maps_row3.append(rd_map[range_low_row3:range_high_row3, :25])
        rd_maps_list_row1.append(rd_maps_row1)
        rd_maps_list_row2.append(rd_maps_row2)
        rd_maps_list_row3.append(rd_maps_row3)
        print("rd shape", rd_map.shape)
        print("rd list shape", np.array(rd_maps_list_row1).shape)

    # Save all data to processed data folder
    # if json_data["is_benchmark"]:
    #     root_data_folder = "../benchmark_dataset"
    # else:
    root_data_folder = "processed_dataset"
        
    prcessed_data_folder = os.path.join(root_data_folder, casename)
    if not os.path.exists(prcessed_data_folder):
        os.makedirs(prcessed_data_folder)
        
    np.save(os.path.join(prcessed_data_folder, 'row1_heatmap.npy'), np.array(row1_heatmap_samples))
    np.save(os.path.join(prcessed_data_folder, 'row2_heatmap.npy'), np.array(row2_heatmap_samples))
    np.save(os.path.join(prcessed_data_folder, 'row3_heatmap.npy'), np.array(row3_heatmap_samples))
    
    np.save(os.path.join(prcessed_data_folder, 'row1_beamformed_phase.npy'), np.array(row1_beamformed_signals_list))
    np.save(os.path.join(prcessed_data_folder, 'row2_beamformed_phase.npy'), np.array(row2_beamformed_signals_list))
    np.save(os.path.join(prcessed_data_folder, 'row3_beamformed_phase.npy'), np.array(row3_beamformed_signals_list))
    
        
    print(np.array(rd_maps_list_row1).shape)
    np.save(os.path.join(prcessed_data_folder, 'row1_rd.npy'), np.array(rd_maps_list_row1))
    np.save(os.path.join(prcessed_data_folder, 'row2_rd.npy'), np.array(rd_maps_list_row2))
    np.save(os.path.join(prcessed_data_folder, 'row3_rd.npy'), np.array(rd_maps_list_row3))

    # Re-save metadata.json
    # Save the dictionary to a JSON file
    file_path = os.path.join(prcessed_data_folder, 'metadata.json')
    with open(file_path, 'w') as file:
        json.dump(json_data, file, indent=4)
        
    np.save(os.path.join(prcessed_data_folder, "config.npy"), np.load(os.path.join(data_path, "config.npy"), allow_pickle=True))

    print(f"Data has been saved to {file_path}")
    source_csv_path = os.path.join(data_path, "respiration_belt.csv")
    target_csv_path = os.path.join(prcessed_data_folder, "respiration_belt.csv")
    if json_data["aligned"]:
        with open(source_csv_path, mode='r', newline='', encoding='utf-8') as source_file:
            with open(target_csv_path, mode='w', newline='', encoding='utf-8') as target_file:
                reader = csv.reader(source_file)
                writer = csv.writer(target_file)
                
                # Copy all rows from the source CSV to the target CSV
                for row in reader:
                    writer.writerow(row)

        print(f"CSV file has been copied to {target_csv_path}")
    return True, "Sucessful"


if __name__ == "__main__":
    # data_path = 'BSENSE/data_preprocessing/preprocessed_dataset'
    # data_gen_one_file(data_path)
    parser = argparse.ArgumentParser(description='Process radar data.')
    parser.add_argument('data_path', type=str, help='Path to the data directory')
    args = parser.parse_args()

    data_gen_one_file(args.data_path)