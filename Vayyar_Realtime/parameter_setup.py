# Author @Sean
import numpy as np
import vtrigU as vtrig
from math import ceil, log
from scipy.constants import c
import os
import time
import datetime

def main():
    for old_data in os.listdir('{}'):
        os.remove(os.path.join('{}',old_data))
    # Radar Setup
    start_freq = 62.0*1000
    stop_freq = 69.0*1000
    num_freq_step = 150
    rbw = 10
    # FFT Bins
    range_Nfft = 512
    angle_Nfft = [64, 64]
    # Doppler Setup
    doppler_window_size = 15
    doppler_Nfft = None
    if doppler_Nfft is None:
        doppler_Nfft = 2**(ceil(log(doppler_window_size,2))+1)
    
    print('Setting up the radar...')
    # initialize the device
    vtrig.Init()
    # set setting structure
    vtrigSettings = vtrig.RecordingSettings(
            vtrig.FrequencyRange(start_freq, # Start Frequency (in MHz)
                                 stop_freq, # Stop  Frequency (in MHz) (66.5 for 5m) (68.0 for 3m)
                                 num_freq_step),      # Number of Frequency Points (Maximum: 150)
            rbw,                           # RBW (in KHz)
            vtrig.VTRIG_U_TXMODE__LOW_RATE  # Tx Mode (LOW: 20 Tx, MED: 10 Tx, HIGH: 4 Tx)
            ) 

    # validate settings
    vtrig.ValidateSettings(vtrigSettings)

    # apply settings
    vtrig.ApplySettings(vtrigSettings)
    print('Done')

    # Calibrating Doppler frame duration
    print('Calibrating Frame Duration for Doppler...')
    frame_duration = []
    for i in range(10):
        data_queue = sorted(os.listdir('./data_queue'))
        start = time.time()
        vtrig.Record()
        rec = vtrig.GetRecordingResult()
        rec_arr = rec2arr(rec)
        pro_arr = rec_arr - rec_arr
        if len(data_queue) >= doppler_window_size*2:
            os.remove(os.path.join('./data_queue',data_queue[0]))
        np.save(f'./data_queue/{datetime.datetime.now().strftime("%m-%d-%Y--%H-%M-%S")}_{time.time_ns()}.npy',pro_arr)
        frame_duration.append(time.time()-start)
    frame_duration = np.mean(frame_duration)
    d = frame_duration
    # d = 0.2 #0.24
    # d = 1/fs
    doppler_freq = np.fft.fftfreq(doppler_Nfft,d)
    doppler_freq = doppler_freq[doppler_freq>=0]

    for old_data in os.listdir('./data_queue'):
        os.remove(os.path.join('./data_queue',old_data))
    print('Done')
    # get antenna pairs and convert to numpy matrix
    TxRxPairs = np.array(vtrig.GetAntennaPairs(vtrigSettings.mode))

    # get used frequencies in Hz
    freq = np.array(vtrig.GetFreqVector_MHz()) * 1e6

    # define constants
    Ts = 1/range_Nfft/(freq[1]-freq[0]+1e-16) # Avoid nan checks
    time_vec = np.linspace(0,Ts*(range_Nfft-1),num=range_Nfft)
    dist_vec = time_vec*(c/2) # distance in meters

    # Parameter Setup
    if angle_Nfft[0] == 64:
        x_offset_shift = -11
        x_ratio = 20/(34.2857)
    elif angle_Nfft[0] == 512: #(Nfft=512)
        x_offset_shift = -90
        x_ratio = 20/30
    else: #(Nfft=512)
        x_offset_shift = 0
        x_ratio = 1

    if angle_Nfft[1] == 64:
        y_offset_shift = 27
        y_ratio = 20/29
    elif angle_Nfft[1] == 512: #(Nfft=512)
        y_offset_shift = 220 
        y_ratio = 20/25
    else: #(Nfft=512)
        y_offset_shift = 0 
        y_ratio = 1

    
    # Data Formation
    AoD_vec = (np.linspace(-90,90,angle_Nfft[0]))*x_ratio
    AoA_vec = (np.linspace(-90,90,angle_Nfft[1]))*y_ratio
    
    
    print("Saving parameters to '{}' ...")
    save_params(
        # Radar Setup
        start_freq = start_freq,
        stop_freq = stop_freq,
        num_freq_step = num_freq_step,
        rbw = rbw,
        # Data Formation
        TxRxPairs = TxRxPairs,
        freq = freq,
        dist_vec = dist_vec,
        AoD_vec = AoD_vec,
        AoA_vec = AoA_vec,
        doppler_freq = doppler_freq,
        # FFT Bins
        range_Nfft = range_Nfft,
        angle_Nfft = angle_Nfft,
        doppler_Nfft = doppler_Nfft,
        # Data Calibration
        x_offset_shift = x_offset_shift,
        y_offset_shift = y_offset_shift,
        x_ratio =  x_ratio,
        y_ratio = y_ratio,
        doppler_window_size = doppler_window_size
    )
    print('Done')
    return

def rec2arr(rec):
    recArr = []
    for key in rec.keys():
        recArr.append(rec[key])
    return np.array(recArr)

def save_params(
        # Radar Setup
        start_freq,
        stop_freq,
        num_freq_step,
        rbw,
        # Data Formation
        TxRxPairs,
        freq,
        dist_vec,
        AoD_vec,
        AoA_vec,
        doppler_freq,
        # FFT Bins
        range_Nfft,
        angle_Nfft,
        doppler_Nfft,
        # Data Calibration
        x_offset_shift,
        y_offset_shift,
        x_ratio,
        y_ratio,
        doppler_window_size
):
    np.save('{}/start_freq.npy', start_freq)
    np.save('{}/stop_freq.npy', stop_freq)
    np.save('{}/num_freq_step.npy', num_freq_step)
    np.save('{}/rbw.npy', rbw)
    np.save('{}/TxRxPairs.npy', TxRxPairs)
    np.save('{}/freq.npy', freq)
    np.save('{}/dist_vec.npy', dist_vec)
    np.save('{}/AoD_vec.npy', AoD_vec)
    np.save('{}/AoA_vec.npy', AoA_vec)
    np.save('{}/doppler_freq.npy', doppler_freq)
    np.save('{}/range_Nfft.npy', range_Nfft)
    np.save('{}/angle_Nfft.npy', angle_Nfft)
    np.save('{}/doppler_Nfft.npy', doppler_Nfft)
    np.save('{}/x_offset_shift.npy', x_offset_shift)
    np.save('{}/y_offset_shift.npy', y_offset_shift)
    np.save('{}/x_ratio.npy', x_ratio)
    np.save('{}/y_ratio.npy', y_ratio)
    np.save('{}/ant_loc.npy',ants_locations())
    np.save('{}/doppler_window_size.npy',doppler_window_size)

    return





def load_params(params_path):
    parameters = {}
    parameters['start_freq'] = np.load('{}/start_freq.npy'.format(params_path))
    parameters['stop_freq'] = np.load('{}/stop_freq.npy'.format(params_path))
    parameters['num_freq_step'] = np.load('{}/num_freq_step.npy'.format(params_path))
    parameters['rbw'] = np.load('{}/rbw.npy'.format(params_path))
    parameters['TxRxPairs'] = np.load('{}/TxRxPairs.npy'.format(params_path))
    parameters['freq'] = np.load('{}/freq.npy'.format(params_path))
    parameters['dist_vec'] = np.load('{}/dist_vec.npy'.format(params_path))
    parameters['AoD_vec'] = np.load('{}/AoD_vec.npy'.format(params_path))
    parameters['AoA_vec'] = np.load('{}/AoA_vec.npy'.format(params_path))
    parameters['doppler_freq'] = np.load('{}/doppler_freq.npy'.format(params_path))
    parameters['range_Nfft'] = np.load('{}/range_Nfft.npy'.format(params_path))
    parameters['angle_Nfft'] = np.load('{}/angle_Nfft.npy'.format(params_path))
    parameters['doppler_Nfft'] = np.load('{}/doppler_Nfft.npy'.format(params_path))
    parameters['x_offset_shift'] = np.load('{}/x_offset_shift.npy'.format(params_path))
    parameters['y_offset_shift'] = np.load('{}/y_offset_shift.npy'.format(params_path))
    parameters['x_ratio'] = np.load('{}/x_ratio.npy'.format(params_path))
    parameters['y_ratio'] = np.load('{}/y_ratio.npy'.format(params_path))
    parameters['ant_loc'] = np.load('{}/ant_loc.npy'.format(params_path))
    parameters['doppler_window_size'] = np.load('{}/doppler_window_size.npy'.format(params_path))

    return parameters

def normalization(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def ants_locations():
    return np.array([[-0.0275, -0.0267, 0], # tx
                     [-0.0253, -0.0267, 0],
                     [-0.0231, -0.0267, 0],
                     [-0.0209, -0.0267, 0],
                     [-0.0187, -0.0267, 0],
                     [-0.0165, -0.0267, 0],
                     [-0.0143, -0.0267, 0],
                     [-0.0122, -0.0267, 0],
                     [-0.0100, -0.0267, 0],
                     [-0.0078, -0.0267, 0],
                     [-0.0056, -0.0267, 0],
                     [-0.0034, -0.0267, 0],
                     [-0.0012, -0.0267, 0],
                     [ 0.0009, -0.0267, 0],
                     [ 0.0031, -0.0267, 0],
                     [ 0.0053, -0.0267, 0],
                     [ 0.0075, -0.0267, 0],
                     [ 0.0097, -0.0267, 0],
                     [ 0.0119, -0.0267, 0],
                     [ 0.0141, -0.0267, 0],
                     [ 0.0274, -0.0133, 0], # rx
                     [ 0.0274, -0.0112, 0],
                     [ 0.0274, -0.0091, 0],
                     [ 0.0274, -0.0070, 0],
                     [ 0.0274, -0.0049, 0],
                     [ 0.0274, -0.0028, 0],
                     [ 0.0274, -0.0007, 0],
                     [ 0.0275,  0.0014, 0],
                     [ 0.0275,  0.0035, 0],
                     [ 0.0275,  0.0056, 0],
                     [ 0.0275,  0.0078, 0],
                     [ 0.0275,  0.0099, 0],
                     [ 0.0275,  0.0120, 0],
                     [ 0.0274,  0.0141, 0],
                     [ 0.0274,  0.0162, 0],
                     [ 0.0275,  0.0183, 0],
                     [ 0.0275,  0.0204, 0],
                     [ 0.0275,  0.0225, 0],
                     [ 0.0275,  0.0246, 0],
                     [ 0.0275,  0.0267, 0]])




if __name__ == '__main__':
    main()