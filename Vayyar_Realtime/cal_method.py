import time
import os
import numpy as np

def calibrate_pro_array(data_queue_folder, cur_folder_name, data_queue, cal_arr, range_Nfft, cur_frame, n_antenna):
    time_window_len = 6
    cal_frame_len = 5
    if time_window_len < cal_frame_len:
        time_window_len = cal_frame_len
    choose_center = False
    if choose_center:
        ants_loc = 'center'
    else:
        ants_loc = 'front'
    # Record the starting time
    start = time.time()
    cal_arr = np.array([cal_arr])
    # cal_arr = np.mean(cal_arr, axis=0)
    cal_frame = np.fft.ifft(cal_arr, axis=2, n=range_Nfft)
    cal_frame = np.mean(cal_frame, axis=0)
    # cal_arr = cal_frame[len(cal_frame)//2]#np.mean(cal_frame, axis=0)
    data_arr = []
    for cur_frame_data in data_queue:
        # cur_frame_data = data # data_queue[cur_frame]
        data_arr.append(np.load(os.path.join(data_queue_folder, cur_folder_name, 'data_queue', cur_frame_data)))
    rec_arr = data_arr[cur_frame]#np.mean(np.array(data_arr), axis=0)
    data_arr = np.stack(data_arr,axis=0)
    cal_frame = data_arr[cur_frame-time_window_len:cur_frame-time_window_len+cal_frame_len]
    cal_frame = np.fft.ifft(cal_frame, n=range_Nfft, axis=2)
    cal_frame = np.mean(cal_frame, axis=0)
    doppler_cal_frame = data_arr[0:cal_frame_len]
    doppler_cal_frame = np.fft.ifft(doppler_cal_frame, n=range_Nfft, axis=2)
    doppler_cal_frame = np.mean(doppler_cal_frame,axis=0)

    # rec_arr = np.load(os.path.join(data_queue_folder,data_queue[cur_frame]))

    # cur_frame_data = data_queue[cur_frame]
    # rec_arr = np.load(os.path.join(data_queue_folder,cur_frame_data))
    pro_arr = rec_arr
    # Extract Certain Number of antennas

    if choose_center:
        center_antenna = 10
        start_antenna = center_antenna - n_antenna//2
        end_antenna = center_antenna + n_antenna//2
    else:
        start_antenna = 0
        end_antenna = n_antenna
    pro_arr = pro_arr.reshape(n_antenna,n_antenna,-1)
    pro_arr = pro_arr[start_antenna:end_antenna,start_antenna:end_antenna,:]
    pro_arr = pro_arr.reshape(-1,150)
    return cal_frame, doppler_cal_frame