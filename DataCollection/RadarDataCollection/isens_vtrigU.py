"""     
    iSens Lab: Step-CW Radar Data Processing with FFT
    Author: Sean Yao 
"""

""" Import Librarys """
from datetime import datetime
import multiprocessing as mp
import os
import sys
from collections import OrderedDict
from math import ceil, log
from os import listdir

import numpy as np
import vtrigU as vtrig
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from pyargus.directionEstimation import *
from scipy.constants import c
from scipy.signal import find_peaks
import time

"""**********************************************************************"""
"""**********************************************************************"""
"""**********************************************************************"""
"""********************** Class: isens_vtrigU ***************************"""
"""**********************************************************************"""
"""**********************************************************************"""
"""**********************************************************************"""
class isens_vtrigU:

    def __init__(
            self, 
            case = None,
            data_folder = './data',
            peak_height = 0.3, 
            enhance = True, 
            enhance_rate = 100, 
            angle_nfft = [512,512],
            range_nfft = None,
            interpolate = True,
            info = False
        ) -> None:
        """ Load Parameters """
        # load setup parameters
        # if case is not None:
        #     if case in listdir(data_folder):
        #         parameter_path = os.path.join(data_folder,case,'constants')
        #         self.freq = np.load(os.path.join(parameter_path,'freq.npy'))
        #         # self.nframes = np.load(os.path.join(parameter_path,'nframes.npy'))
        #         self.TxRxPairs = np.load(os.path.join(parameter_path,'TxRxPairs.npy'))
        #     else:
        #         print(f'Directory: {case} does not exist! Aborting the program...')
        #         print('')
        #         sys.exit()
        # else:
        #     current_dir = os.path.dirname(os.path.abspath(__file__))
        #     self.freq = np.load(os.path.join(current_dir, './constants/freq.npy'))
        #     self.nframes = np.load(os.path.join(current_dir, './constants/nframes.npy'))
        #     self.txRxPairs = np.load(os.path.join(current_dir, './constants/TxRxPairs.npy'))

        self.ants_locations = self._ants_locations()[:,:-1]
        self.data_folder = data_folder
        # if range_nfft is None:
        #     self.range_nfft = 2**(ceil(log(self.freq.shape[0],2))+1)
        # else:
        #     self.range_nfft = range_nfft
        # Make sure it can run
        # self.nfft = 2**(ceil(log(self.freq.shape[0],2))+1)
        # End change
        self.center_ant = 10
        self.peak_height = peak_height
        self.enhance_rate = enhance_rate
        self.enhance = enhance
        self.interpolate = interpolate
        # self.dist_vec = self.compute_dist_vec()
        self.d = 0.5
        self.aoa_offset = 90-83.83561643835615-6.34050881
        self.aod_offset = 90-50.0196#131.5068493
        self.BW = 7e9
        self.Rres = c/(2*self.BW)
        
        self.angle_nfft = angle_nfft
        self.AoD_vec = np.linspace(0,180,self.angle_nfft[0])
        self.AoA_vec = np.linspace(0,180, self.angle_nfft[1])
        # self.N_freq = self.freq.shape[0]
        
        if info:
            print(f'Freq Points: {self.freq.shape[0]} ')
            print(f'TxRxPairs Shape: {self.TxRxPairs.shape}')
            print(f'nfft = {self.nfft}')
            print(f'Number of Recorded Frames: {self.nframes}')
            print('')


    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """********************** Other Helper Functions ************************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""
    # Return Vayyar VtrigU Radar Antennas' 3D Locations
    def _ants_locations(self):
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
    
    # Plot the Antenna Array
    def plot_antennas(self):
        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.plot(self.ants_locations[ 0:20,0],self.ants_locations[ 0:20,1],'^')
        plt.plot(self.ants_locations[20:40,0],self.ants_locations[20:40,1],'^')
        # plt.plot(np.mean(antsLocations[ 0:20,0]),np.mean(antsLocations[20:40,1]),'*')
        plt.title('Uniform Linear Array')
        plt.legend(['Tx Antenna','Rx Antenna'])
        plt.grid()
        plt.axis('scaled')
        plt.xlim([-0.04,0.04])
        plt.ylim([-0.04,0.04])
        plt.show(block=True)

    # Compute Distance Vector
    def compute_dist_vec(self, nfft=None):
        if nfft == None:
            nfft = self.nfft
        Ts = 1/nfft/(self.freq[1]-self.freq[0]+1e-16) # Avoid nan checks
        self.time_vec = np.linspace(0,Ts*(nfft-1),num=nfft)
        return self.time_vec*(c/2) # distance in meters
    
    # Normalize the signal to the range [0,1]
    def normalization(self, x):
        return (x - np.min(x))/(np.max(x)-np.min(x))

    # Load collected Data
    def load_data(self, data_folder='./data', case = 'test/', scenario='move_z', return_path=False, info=False):
        # specify data path components

        data_path = os.path.join(self.data_folder, case, "")

        if scenario in listdir(data_path):
            raw_data = 'recording.npy'
            cal_data = 'calibration.npy'

            # combine data paths
            raw_path = os.path.join(data_path, scenario, raw_data)
            cal_path = os.path.join(data_path, scenario, cal_data)
            
            # processed_path = data_path + processed_data

            # load data
            recArr = np.load(raw_path)
            calArr = np.load(cal_path)
            if info:
                print('Current scenario: ' + scenario)
                print('')
                print(f'calArr Shape: {calArr.shape}')
                print(f'recArr Shape: {recArr.shape}')
                print('')
                print('recArr Channels: (frame, Tx*Rx, freqs)')
                print('')

        else:
            print(f'Scenario: {scenario} does not exist!')
            print('')
            sys.exit()
        
        if return_path:
            return calArr, recArr, os.path.join(data_path, scenario, '')
        else:
            return calArr, recArr
    
    # Plot 2D heatmaps
    def heatmap_2D(self, arr, fnum):
        fig, ax = plt.subplots(figsize=(8,8))
        extent = [-90, 90, 0, np.max(self.dist_vec)]
        ax.imshow(np.abs(arr[fnum].T),origin='lower', interpolation='nearest', aspect='auto', extent=extent)
        ax.set_title("2-D Heat Map")
        ax.set_xlabel("Angle [deg]")
        ax.set_ylabel("Range [m]")


    def interactive_heatmap_2d(self, arr, title='2-D Heat Map', method='music_aoa'):
        init_fnum = 0
        cur_frame = np.abs(arr[init_fnum].T)
        fig, ax = plt.subplots(figsize=(10,8))
        plt.ion()

        if method=='music_aoa':
            extent = [0+self.aoa_offset, 180+self.aoa_offset, 0, np.max(self.dist_vec)]
        elif method=='music_aod':
            extent = [-41, 41, 0, np.max(self.dist_vec)]

        else:
            extent = [-90, 90, 0, np.max(self.dist_vec)] 
        img = ax.imshow(self.normalization(np.abs(arr[init_fnum,:,:].T)),origin='lower', interpolation=None, aspect='auto', extent=extent)

        fig.colorbar(img)
        ax.set_title(title)
        ax.set_xlabel("Angle [deg]")
        ax.set_ylabel("Range [m]")
        ax.grid()

        fig.subplots_adjust(left=0.25, bottom=0.25)
        axframe = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        frame_slider = Slider(
            ax=axframe,
            label='frame',
            valmin=0,
            valmax=arr.shape[0]-1,
            valinit=init_fnum,
            valstep=1,
        )
        def update(val):
            img.set_data(self.normalization(np.abs(arr[int(frame_slider.val)].T)))
            fig.canvas.draw_idle()
        frame_slider.on_changed(update)
        resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')
        def reset(event):
            frame_slider.reset()

        button.on_clicked(reset)
        plt.show(block=True)
    

    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """***************** RADAR SCANNING Helper Functions ********************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""

    # Convert the recorded dictionary to array
    def rec2arr(self, rec):
        recArr = []
        for key in rec.keys():
            recArr.append(rec[key])
        return np.array(recArr)

    # Scanning Setup
    def scan_setup(self, 
                   start_freq = 62.0*1000,
                   stop_freq = 69.0*1000,
                   n_freq = 150,
                   rbw = 80.0,
                   scan_profile = vtrig.VTRIG_U_TXMODE__LOW_RATE
                  ):
        
        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.n_freq = n_freq
        self.rbw = rbw
        self.scan_profile = scan_profile

        # initialize the device
        vtrig.Init()

        # set setting structure
        vtrigSettings = vtrig.RecordingSettings(
                vtrig.FrequencyRange(self.start_freq, # Start Frequency (in MHz)
                                     self.stop_freq, # Stop  Frequency (in MHz) (66.5 for 5m) (68.0 for 3m)
                                     self.n_freq),      # Number of Frequency Points (Maximum: 150)
                self.rbw,                           # RBW (in KHz)
                self.scan_profile  # Tx Mode (LOW: 20 Tx, MED: 10 Tx, HIGH: 4 Tx)
                ) 

        # validate settings
        vtrig.ValidateSettings(vtrigSettings)

        # apply settings
        vtrig.ApplySettings(vtrigSettings)

        # get antenna pairs and convert to numpy matrix
        self.txRxPairs = np.array(vtrig.GetAntennaPairs(vtrigSettings.mode))

        # get used frequencies in Hz
        self.freq = np.array(vtrig.GetFreqVector_MHz()) * 1e6

        # define constants
        self.nfft = 2**(ceil(log(self.freq.shape[0],2))+1)
        self.dist_vec = self.compute_dist_vec()

    # Record the background data for calibration usage
    def scan_calibration(self, nrecs=10):
        # Record the calibration frames
        print("calibrating...")
        calFrame = []
        for i in range(nrecs):
            vtrig.Record()
            rec = vtrig.GetRecordingResult()
            recArr = self.rec2arr(rec)
            calFrame.append(recArr)
        calFrame = np.array(calFrame)
        print("calibration matrix collected!")
        print()
        return calFrame

    # scan the data
    def scan_data(self, nframes=100, print_progress=True):
        self.nframes = nframes
        if print_progress:
            print("recording...")
        recArrs = []
        for i in range(self.nframes):
            # write_read(str(motion_stage[i]))
            vtrig.Record()
            rec = vtrig.GetRecordingResult()
            recArrs.append(self.rec2arr(rec))
        recArrs = np.array(recArrs)
        if print_progress:
            print("record done!")
            print()
        return recArrs
    
    # scan the data
    def test_scan_data(self, nframes=100, print_progress=True):
        self.nframes = nframes
        if print_progress:
            print("recording...")
        recArrs = []
        for i in range(self.nframes):
            time = time.time()
            vtrig.Record()
            rec = vtrig.GetRecordingResult()
            recArrs.append(rec)
        recArrs =  np.array([self.rec2arr(rec) for rec in recArrs])
        if print_progress:
            print("record done!")
            print()
        return recArrs
    
    def yield_data(self, nframes=100, print_progress=True):
        self.nframes = nframes
        if print_progress:
            print("recording...")
        for i in range(self.nframes):
            vtrig.Record()
            rec = vtrig.GetRecordingResult()
            yield rec
    
    # scan the data
    def scan_save_data(self, cal_arr, data_dir, folder_name, nframes=100, print_progress=True, is_calibration=True):
        self.nframes = nframes
        if print_progress:
            print("recording...")
        recArrs = []
        for i in range(self.nframes):
            # write_read(str(motion_stage[i]))
            vtrig.Record()
            # Three times
            rec = vtrig.GetRecordingResult()
            # recArrs.append(self.rec2arr(rec))
            rec_arr = self.rec2arr(rec)
            if is_calibration:
                pro_arr = rec_arr - cal_arr
            else:
                pro_arr = rec_arr
            # Fix it
            np.save(f'{os.path.join(data_dir, folder_name)}/data_queue/{datetime.now().strftime("%m-%d-%Y--%H-%M-%S")}_{time.time_ns()}.npy',pro_arr)
            recArrs = np.array(rec_arr)
        if print_progress:
            print("record done!")
            print()
        return recArrs
    
    # dealing with directory creations
    def dname_process(self, case, scenario):
        now = datetime.now()
        current_date = now.strftime("%m%d%Y")
        current_time = now.strftime("%H%M%S")

        if case == None or case not in listdir(self.data_folder):
            case == f'test{current_date}'
            if case not in listdir(self.data_folder):
                print(f"The case does not exist in the current data directory, creating a new directory with name: {case} ...")
            else:
                print(f'Creating a new directory with name: {case} ...')
            os.mkdir(os.path.join(self.data_folder,case))
    
        if scenario == None:
            scenario = input("Please Enter the Current Recording's Scenario Name")
            print('')
        if scenario not in listdir(os.path.join(self.data_folder,case)):
            scenario = f'{scenario}_{current_time}'
            os.mkdir(os.path.join(self.data_folder,case,scenario))

        current_path = os.path.join(self.data_folder,case,scenario,'')
        return current_path
    
    # Scanning pipeline by putting all the previous helper functions together
    def scan_pipeline(self, 
                      case=None, 
                      scenario=None, 
                      start_freq = 62.0*1000,
                      stop_freq = 69.0*1000,
                      n_freq = 150,
                      rbw = 80.0,
                      scan_profile = vtrig.VTRIG_U_TXMODE__LOW_RATE,
                      cal_nrecs=10,
                      rec_nframes=100
                     ):
        
        # parameter setup
        self.scan_setup(start_freq=start_freq,
                        stop_freq=stop_freq,
                        n_freq=n_freq,
                        rbw=rbw,
                        scan_profile=scan_profile
                       )
        
        input('Click Enter to record the background data...')
        print('')
        # scan the calibration frames
        cal_Arr = self.scan_calibration(nrecs=cal_nrecs)

        input('Click Enter to record the data...')
        print('')
        # scan the data
        rec_Arr = self.scan_data(nframes=rec_nframes)

        # setup the directories
        current_path = self.dname_process(case, scenario)

        # save both data
        np.save(os.path.join(current_path,'calibration.npy'),cal_Arr)
        np.save(os.path.join(current_path,'recording.npy'),rec_Arr)

        return case, scenario

    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """****************** ToF Processing Helper Functions *******************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""

    # Compute the range profile with ifft and then take the norm
    def compute_tof_ifft(self, X, nfft=512):
        if len(X.shape) == 3:
            ifft_axis = 2
            norm_axis = 1
        elif len(X.shape) == 2:
            ifft_axis = 1
            norm_axis = 0
        else:
            return np.linalg.norm(np.fft.ifft(X,nfft).reshape(-1,1),axis=1)
        x = np.fft.ifft(X,nfft,ifft_axis)
        return np.linalg.norm(x,axis=norm_axis)    

    # Pipeline for computing the range profile with the previous functions
    def range_pipeline(self, case='test/', scenario='move_z', cal_method=0, plot=False, nfft=None):
        if nfft == None:
            nfft = self.nfft
        # Load Data
        calArr, recArr = self.load_data(scenario=scenario, case=case)
        # Calibrate Data
        if cal_method == None:
            cal_methods = [0,1]
        else:
            cal_methods = [cal_method]
        tofs = []
        for cal_method in cal_methods:
            proArr = self.calibration(calArr,recArr,cal_method)
            # Compute Range Profile
            tof = self.compute_tof_ifft(proArr, nfft=nfft).T
            tofs.append(tof.T)
            if plot:
                # plot the range profile vs frame
                extent = [0, 99, 0, np.max(self.dist_vec)]
                plt.figure(figsize=(10,5))
                plt.imshow(self.normalization(tof),origin='lower', interpolation='nearest', aspect='auto', extent=extent)
                plt.colorbar()
                plt.title('Frame vs Distance')
                plt.xlabel('Frame')
                plt.ylabel('Distance [m]')
                plt.show(block=True)
        return np.squeeze(np.array(tofs))


    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """**************** SIGNAL PROCESSING Helper Functions ******************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""

    # Calibraiton fucntions for background subtraction & interperiodic subtraction
    def calibration(self, calArr, recArr, method = 0):
        """
        calibration method:
            0 - subtract the pre-collected background noise
            1 - subtract each frame by the previous frame
        """
        if len(calArr.shape) > 2:
            calFrame = np.mean(calArr,axis=0)
        else:
            calFrame = calArr

        if method == 0:
            proArr = recArr - calFrame
        elif method == 1:
            proArr = recArr[1:recArr.shape[0],:,:] - recArr[0:recArr.shape[0]-1,:,:]
            
        return proArr

    def interpolation(self, signal):
        interpolated_signal = np.zeros([signal.shape[0],signal.shape[1],signal.shape[2]*2],dtype='complex')
        interpolated_signal[:,:,::2] = signal
        for i in np.arange(interpolated_signal.shape[2])[1::2]:
            left = 0
            right = 0
            if i-1 >= 0 and i-1 < interpolated_signal.shape[2]:
                left = interpolated_signal[:,:,i-1]
            if i+1 < interpolated_signal.shape[2] and i+1 >= 0:
                right = interpolated_signal[:,:,i+1]
            interpolated_signal[:,:,i] = (left+right)/2
        interpolated_signal = interpolated_signal[:,:,:-1]
        # Adjust other parameters correspondingly
        self.nfft = 2**(ceil(log(interpolated_signal.shape[2],2))+1)
        xp = np.arange(len(self.dist_vec))
        fp = self.dist_vec
        x = np.arange(self.nfft)
        self.dist_vec = np.interp(x,xp,fp)
        return interpolated_signal


    def enhance_target(self, signal, plot=True):
        if len(signal.shape) == 4:
            for i in range(signal.shape[0]):
                peaks = find_peaks(self.normalization(np.abs(signal[i,self.center_ant,self.center_ant,:])),height=self.peak_height)[0]
                if plot:
                    if i == 50:
                        plt.figure()
                        plt.plot(self.dist_vec,self.normalization(np.abs(signal[i,self.center_ant,self.center_ant,:])))
                        plt.plot(self.dist_vec[peaks],self.normalization(np.abs(signal[i,self.center_ant,self.center_ant,:]))[peaks],'o')
                        plt.xlabel('Distance [m]')
                        plt.ylabel('Normalized Magnitude')
                        plt.title('Peak extraction')
                        plt.show(block=True)
                        print(f'Enhance Peak at: {self.dist_vec[peaks]}')
                signal[i,:,:,peaks] *= self.enhance_rate
        elif len(signal.shape) == 2:
                peaks = find_peaks(self.normalization(np.abs(signal[self.center_ant,:])),height=self.peak_height)[0]
                if plot:
                    plt.figure()
                    plt.plot(self.dist_vec,self.normalization(np.abs(signal[self.center_ant,:])))
                    plt.plot(self.dist_vec[peaks],self.normalization(np.abs(signal[self.center_ant,:]))[peaks],'o')

                    plt.xlabel('Distance [m]')
                    plt.ylabel('Normalized Magnitude')
                    plt.title('Peak extraction')
                    plt.show(block=True)

                print(f'Enhance Peak at: {self.dist_vec[peaks]}')
                signal[:,peaks] *= self.enhance_rate
        return signal, peaks
    
    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """***************** FFT PROCESSING Helper Functions ********************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""

    # Compute Angle FFT from the range profile for estimating AoA
    def compute_aoa_fft(self, X):
        print('Computing AoA...')
        print('')
        if self.interpolate:
            X = self.interpolation(X)
        x = np.fft.ifft(X,self.nfft,2).reshape(X.shape[0],20,20,-1)
        if self.enhance:
            x = self.enhance_target(x)
        aoa = np.fft.fft(x[:,self.center_ant,:,:],self.nfft,axis=1) # extract the center tx data and do the processing
        aoa = np.fft.fftshift(aoa,axes=1)
        return aoa

    # Compute Angle FFT from the range profile for estimating AoD
    def compute_aod_fft(self, X):
        print('Computing AoD...')
        print('')
        if self.interpolate:
            X = self.interpolation(X)
        x = np.fft.ifft(X,self.nfft,2).reshape(X.shape[0],20,20,-1)
        plt.figure()
        plt.plot(self.dist_vec,self.normalization(np.abs(x[50,10,10,:])))
        plt.xlabel('Distance [m]')
        plt.ylabel('Normalized Magnitude')
        plt.title('Peak extraction')
        plt.show(block=True)
        if self.enhance:
            x = self.enhance_target(x)
        aod = np.fft.fft(x[:,:,self.center_ant,:],self.nfft,axis=1) # extract the center rx data and do the processing
        aod = np.fft.fftshift(aod,axes=1)
        return aod
    
    # FFT Processing pipeline by putting all the previous helper functions together
    def fft_pipeline(self, case='test/', scenario='move_z', cal_method=0, all_case=False, plot=None, test_mode=False, show_ants=False):
        if show_ants:
            self.plot_antennas()
        if all_case == False:
            # Load Data
            calArr, recArr, path = self.load_data(scenario=scenario, case=case, return_path=True)
            # Calibrate Data
            proArr = self.calibration(calArr,recArr,cal_method)
            # Compute AoA & ToF
            if f"fft_aoa_1frame_cal{cal_method}.npy" not in listdir(path):
                aoaArr = self.compute_aoa_fft(proArr)
                np.save(path+f"fft_aoa_1frame_cal{cal_method}.npy",aoaArr)
            else:
                if not test_mode:
                    aoaArr = np.load(path+f"fft_aoa_1frame_cal{cal_method}.npy")
                else:
                    aoaArr = self.compute_aoa_fft(proArr)

            
            # Compute AoD & ToF
            if f"fft_aod_1frame_cal{cal_method}.npy" not in listdir(path):
                aodArr = self.compute_aod_fft(proArr)
                np.save(path+f"fft_aod_1frame_cal{cal_method}.npy",aodArr)
            else:
                if not test_mode:
                    aodArr = np.load(path+f"fft_aod_1frame_cal{cal_method}.npy")
                else:
                    aodArr = self.compute_aod_fft(proArr)

            if plot:
                print('Interactive 2D heatmap displaying, please hit interrupt to stop displaying...')
            if plot == 'aoa':
                if self.interpolate:
                    self.interactive_heatmap_2d(aoaArr[:,:,0::2], 'AoA 2D Heatmap')
                else:
                    self.interactive_heatmap_2d(aoaArr, 'AoA 2D Heatmap')
            if plot == 'aod':
                if self.interpolate:
                    self.interactive_heatmap_2d(aodArr[:,:,0::2], 'AoD 2D Heatmap')
                else:
                    self.interactive_heatmap_2d(aodArr, 'AoD 2D Heatmap')
            if plot == 'both':
                if self.interpolate:
                    self.interactive_heatmap_2d(aoaArr[:,:,0::2], 'AoA 2D Heatmap')
                    self.interactive_heatmap_2d(aodArr[:,:,0::2], 'AoD 2D Heatmap')
                else:
                    self.interactive_heatmap_2d(aoaArr, 'AoA 2D Heatmap')
                    self.interactive_heatmap_2d(aodArr, 'AoD 2D Heatmap')
        else:
            for scenario in listdir('./data/' + case):
                if scenario == 'constants':
                    continue
                # Load Data
                calArr, recArr, path = self.load_data(scenario=scenario, case=case, return_path=True)
                print(f'Path: {path}')
                print('')
                # Calibrate Data
                for i in [0]:
                    print(f'Current Calibration Method: {i}')
                    print('')
                    proArr = self.calibration(calArr,recArr,i)
                    # Compute AoA & ToF
                    if f"fft_aoa_1frame_cal{i}.npy" not in listdir(path):
                        aoaArr = self.compute_aoa_fft(proArr)
                        np.save(path+f"fft_aoa_1frame_cal{i}.npy",aoaArr)
                    else:
                        aoaArr = np.load(path+f"fft_aoa_1frame_cal{i}.npy")
                    
                    # Compute AoD & ToF
                    if f"fft_aod_1frame_cal{i}.npy" not in listdir(path):
                        aodArr = self.compute_aod_fft(proArr)
                        np.save(path+f"fft_aod_1frame_cal{i}.npy",aodArr)
                    else:
                        aodArr = np.load(path+f"fft_aod_1frame_cal{i}.npy")
                    
                    if plot:
                        print('Interactive 2D heatmap displaying, please hit interrupt to stop displaying...')
                    if plot == 'aoa':
                        self.interactive_heatmap_2d(aoaArr, 'AoA 2D Heatmap')
                    if plot == 'aod':
                        self.interactive_heatmap_2d(aodArr, 'AoD 2D Heatmap')
                    if plot == 'both':
                        self.interactive_heatmap_2d(aoaArr, 'AoA 2D Heatmap')
                        self.interactive_heatmap_2d(aodArr, 'AoD 2D Heatmap')
    
    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """***************** MUSIC PROCESSING Helper Functions ******************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""

    # MUSIC simulation with user-specified target directions
    def music_sim_signal(self, thetas = [50, 80, 110]):
        # Number of antenna elements
        M = 20
        # number of samples
        N = self.nfft
        # Interelement spacing is half-wavelength
        d= 0.5
        a_list = []
        for theta in thetas:
            a_list.append(np.exp(np.arange(0,M,1)*1j*2*np.pi*d*np.cos(np.deg2rad(theta))))
        # Generate multichannel test signal 
        soi = np.random.normal(0,1,N)  # Signal of Interest
        soi_matrix = np.zeros(np.size(np.outer( soi, a_list[0])))
        for a in a_list:
            soi_matrix  += np.outer( soi, a)
        soi_matrix = soi_matrix.T
        # Generate multichannel uncorrelated noise
        noise = np.random.normal(0,np.sqrt(10**-1),(M,N))
        # Create received signal array
        rec_signal = soi_matrix + noise
        return rec_signal 
    
    # Compute AoA by using MUSIC on the range profile, bin by bin, for a single frame
    def compute_aoa_music_single_frame(self, rec_signal, signal_dimension=3, smoothing=True, plot=False, n_rx=20): 
        if smoothing:
            # Calculate the forward-backward spatially smoothed correlation matrix
            R = spatial_smoothing(rec_signal.T, P=n_rx, direction="forward-backward")
        else:
            # Estimating the spatial correlation matrix without spatial smoothing
            R = corr_matrix_estimate(rec_signal.T, imp="mem_eff")

        # Regenerate the scanning vector for the sub-array
        array_alignment = np.arange(0, n_rx, 1) * self.d
        incident_angles = self.angle_vec
        scanning_vectors = gen_ula_scanning_vectors(array_alignment, incident_angles)

        # Estimate DOA 
        aoa = DOA_MUSIC(R, scanning_vectors, signal_dimension=signal_dimension)
        # print(f'Max Peak at: {incident_angles[np.argmax(aoa)]-90+self.aoa_offset} deg')
        # print()
        
        if plot:
            # Get matplotlib axes object
            plt.figure()
            axes = plt.axes()
            
            # Plot results on the same fiugre
            DOA_plot(aoa, incident_angles+self.aoa_offset-90, log_scale_min = -50, axes=axes, alias_highlight=False)

            axes.legend(["MUSIC"])

            # Mark nominal incident angles
            # axes.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_1)
            plt.show()
        return aoa

    # Compute AoA by using MUSIC on the range profile, using all of the range bins simultaneously, frame by frame
    def compute_aoa_music(self, X, n_rx=20, signal_dimension=3, smoothing=True, plot=False): 
        # Regenerate the scanning vector for the sub-array
        aoa = []
        array_alignment = np.arange(0, n_rx, 1) * self.d
        incident_angles= self.angle_vec
        scanning_vectors = gen_ula_scanning_vectors(array_alignment, incident_angles)
        for frame in range(X.shape[0]):
            rec_signal = X.reshape((X.shape[0],20,20,-1))[frame,self.center_ant,:,:]
            rec_signal = np.fft.ifft(rec_signal,axis=1,n=self.range_nfft)
            if smoothing:
                # Calculate the forward-backward spatially smotthed correlation matrix
                R = spatial_smoothing(rec_signal.T, P=n_rx, direction="forward-backward")
            else:
                # Estimating the spatial correlation matrix without spatial smoothing
                R = corr_matrix_estimate(rec_signal.T, imp="mem_eff")
            # Estimate DOA 
            aoa.append(DOA_MUSIC(R, scanning_vectors, signal_dimension=signal_dimension))

        return np.array(aoa)
    
    # Compute AoD by using MUSIC on the range profile, bin by bin, for a single frame
    def compute_aod_music_single_frame(self, rec_signal, signal_dimension=3, smoothing=True, plot=False, n_tx=20):
        if smoothing:
            # Calculate the forward-backward spatially smotthed correlation matrix
            R = spatial_smoothing(rec_signal.T, P=n_tx, direction="forward-backward")
        else:
            # Estimating the spatial correlation matrix without spatial smoothing
            R = corr_matrix_estimate(rec_signal.T, imp="mem_eff")

        # Regenerate the scanning vector for the sub-array
        array_alignment = np.arange(0, n_tx, 1) * self.d
        incident_angles= self.angle_vec
        scanning_vectors = gen_ula_scanning_vectors(array_alignment, incident_angles)

        # Estimate DOA 

        # print('Computing AoD...')
        aod = DOA_MUSIC(R, scanning_vectors, signal_dimension=signal_dimension)
        # print('...Done')
        # print()
        # aod_angles = incident_angles-90+self.aod_offset
        # print(f'Max Peak at: {aod_angles[np.where(np.abs(aod_angles) <= 41)][np.argmax(aod[np.where(np.abs(aod_angles) <= 41)])]} deg')
        # print()

        if plot:
            # Get matplotlib axes object
            plt.figure()
            axes = plt.axes()
            
            # Plot results on the same fiugre
            DOA_plot(aod, incident_angles-90+self.aod_offset, log_scale_min = -50, axes=axes, alias_highlight=False)

            axes.legend(["MUSIC"])

            # Mark nominal incident angles
            # axes.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_1)
            plt.show()
        return aod

    # Compute AoD by using MUSIC on the range profile, using all of the range bins simultaneously, frame by frame
    # TODO
    # def compute_aoa_music(self, X, n_rx=20, signal_dimension=3, smoothing=True, plot=False): 

    # MUSIC Processing for AoA single frame by putting all the previous helper functions together
    def music_aoa_single_frame_pipeline(self, case='test/', scenario='move_z', simulation=False, cal_method=0, signal_dimension=3, smoothing=True, plot_aoa_spectrum=False):
        if simulation:
            rec_signal = self.music_sim_signal()
            aoa = self.compute_aoa_music_single_frame(rec_signal=rec_signal, signal_dimension=signal_dimension, smoothing=smoothing, plot=plot_aoa_spectrum)

        else:
            # Load Data
            calArr, recArr = self.load_data(scenario=scenario, case=case)
            # Calibrate Data
            proArr = self.calibration(calArr,recArr,cal_method).reshape(100,20,20,-1)
        
            range_profile = np.real(np.fft.ifft(proArr[50,self.center_ant,:,:],n=self.nfft, axis=1))
            # range_profile = np.linalg.norm(range_profile,axis=0)
            range_profile[:,np.where(self.dist_vec>2.5)] = 0
            range_profile[:,np.where(self.dist_vec<0.3)] = 0
            if self.enhance:
                range_profile, peaks = self.enhance_target(range_profile)
            else:
                _, peaks = self.enhance_target(range_profile)

            # Compute AoA
            aoa = []
            pool = mp.Pool() # Threadpool for multi-threading
            outputs = []
            for range_bin in range(range_profile.shape[1]):
                rec_signal = range_profile[:,range_bin].reshape((-1,1))
                # print(rec_signal.shape)
                # aoa.append(self.compute_aoa_music_single_frame(rec_signal=rec_signal, signal_dimension=signal_dimension, smoothing=smoothing, plot=plot_aoa_spectrum))
                outputs.append(pool.apply_async(self.compute_aoa_music_single_frame,(rec_signal,signal_dimension,smoothing,plot_aoa_spectrum)))
            pool.close()
            aoa = [res.get() for res in outputs]
            heat_map = np.array(aoa)
            heat_map[peaks,:] *= 1000
            heat_map = self.normalization(np.abs(heat_map))
            angle_peaks = []
            for i in range(len(peaks)):
                angle_peaks.append(find_peaks(np.squeeze(self.normalization(np.abs(heat_map[peaks[i],:]))),height=self.peak_height)[0])

            # angle_peaks = np.squeeze(angle_peaks)
            # aoa_angles = self.angle_vec-90+self.aoa_offset

            # print(f'Target at: {self.dist_vec[peaks]} m & {aoa_angles[angle_peaks]} deg in y direction')
            # print(np.where(np.abs(heat_map)==np.max(np.abs(heat_map))))
            plt.figure()
            extent = [-90+self.aoa_offset, 90+self.aoa_offset, 0, np.max(self.dist_vec)]
            plt.imshow(heat_map,origin='lower', interpolation=None, aspect='auto', extent=extent)

            plt.colorbar()
            plt.title('MUSIC AoA')
            plt.xlabel("Angle [deg]")
            plt.ylabel("Range [m]")
            plt.grid()
            plt.show()

    # MUSIC Processing for AoA single frame by putting all the previous helper functions together
    def music_aod_single_frame_pipeline(self, case='test/', scenario='move_z', proArr=None, simulation=False, cal_method=0, signal_dimension=3, smoothing=True, plot_aod_spectrum=False, plot_heatmap=True, plot_peaks=True, frame=50):
        if simulation:
            rec_signal = self.music_sim_signal()
            aod = self.compute_aod_music_single_frame(rec_signal=rec_signal, signal_dimension=signal_dimension, smoothing=smoothing, plot=plot_aod_spectrum)
        else:
            if proArr is None:
                # Load Data
                calArr, recArr = self.load_data(scenario=scenario, case=case)
                # Calibrate Data
                proArr = self.calibration(calArr,recArr,cal_method).reshape(100,20,20,-1)
            rec_signal = np.fft.ifft(proArr[frame,:,self.center_ant,:],axis=1,n=self.range_nfft)
        
            range_profile = np.real(np.fft.ifft(proArr[frame,:,self.center_ant,:],n=self.range_nfft, axis=1))
            range_profile[:,np.where(self.dist_vec>2.5)] = 0
            range_profile[:,np.where(self.dist_vec<0.3)] = 0
            if self.enhance:
                range_profile, peaks = self.enhance_target(range_profile, plot=plot_peaks)
            else:
                _, peaks = self.enhance_target(range_profile, plot=plot_peaks)
                
            aod = []
            pool = mp.Pool()
            outputs = []
            for range_bin in range(range_profile.shape[1]):

                if range_bin in peaks:
                    rec_signal = range_profile[:,range_bin].reshape((-1,1))
                    # print(rec_signal.shape)
                    # aod.append(self.compute_aod_music_single_frame(rec_signal=rec_signal, signal_dimension=signal_dimension, smoothing=smoothing, plot=plot_aod_spectrum))
                    # outputs.append(pool.apply_async(self.compute_aod_music_single_frame,(rec_signal,signal_dimension,smoothing,plot_aod_spectrum)))
                    output = pool.apply_async(self.compute_aod_music_single_frame,(rec_signal,signal_dimension,smoothing,plot_aod_spectrum))
                    aod.append(output.get())
                else:
                    aod.append(np.zeros(self.range_nfft))
            pool.close()
            # aod = [res.get() for res in outputs]

            heat_map = np.array(aod)
            aod_angles = self.angle_vec-90+self.aod_offset
            # print(np.where(np.abs(aod_angles) <= 41) )
            heat_map = np.squeeze(heat_map[:,np.where(np.abs(aod_angles) <= 41)])

            heat_map[peaks,:] *= 1000

            heat_map = self.normalization(np.abs(heat_map))
            angle_peaks = []
            for i in range(len(peaks)):
                angle_peaks.append(find_peaks(np.squeeze(self.normalization(np.abs(heat_map[peaks[i],:]))),height=0.8)[0])
            angle_peaks = np.squeeze(angle_peaks)
            try:
                print(f'Target at: {self.dist_vec[peaks]} m & {aod_angles[angle_peaks]+9.15854324853229} deg in x direction')
            except:
                print('processing done')
            print()
            if plot_heatmap:
                plt.figure()
                plt.plot(aod_angles[np.where(np.abs(aod_angles) <= 41)],np.squeeze(self.normalization(np.abs(heat_map[peaks[i],:]))))
                plt.plot(aod_angles[np.where(np.abs(aod_angles) <= 41)][angle_peaks],np.squeeze(self.normalization(np.abs(heat_map[peaks[i],:])))[angle_peaks],'o')
                plt.show()
                
                plt.figure()
                # extent = [-90+self.aod_offset, 90+self.aod_offset, 0, np.max(self.dist_vec)]
                extent = [-41, 41, 0, np.max(self.dist_vec)]
                plt.imshow(heat_map,origin='lower', interpolation=None, aspect='auto', extent=extent)
                plt.colorbar()
                plt.title('MUSIC AoD')
                plt.xlabel("Angle [deg]")
                plt.ylabel("Range [m]")
                plt.grid()
                plt.show()
            return heat_map

    # MUSIC Processing for either AoA or AoD for all frames by putting all the previous helper functions together
    def music_pipeline(self, case='test/', scenario='move_z', mode='aoa', simulation=False, cal_method=0, signal_dimension=3, smoothing=True, plot_spectrum=False):
        if mode == 'aoa':
            fname = f'aoa_music_SOI_{signal_dimension}_ph_{self.peak_height}.npy'
        else:
            fname = f'aod_music_SOI_{signal_dimension}_ph_{self.peak_height}.npy'

        dir_path = os.path.join(self.data_folder,case,scenario,'')
        if fname in listdir(dir_path):
            heatmaps = np.load(os.path.join(dir_path,fname))
        else:
            # Load Data
            calArr, recArr = self.load_data(scenario=scenario, case=case)
            # Calibrate Data
            proArr = self.calibration(calArr,recArr,cal_method).reshape(100,20,20,-1)

            heatmaps = []
            for i in range(self.nframes):
                print(f'frame: {i}')
                if mode == 'aoa':
                    heatmaps.append(self.music_aoa_single_frame_pipeline(case, scenario, proArr, simulation, cal_method, signal_dimension, smoothing, plot_spectrum, False, False, i).T)
                else:
                    heatmaps.append(self.music_aod_single_frame_pipeline(case, scenario, proArr, simulation, cal_method, signal_dimension, smoothing, plot_spectrum, False, False, i).T)
            heatmaps = np.stack(heatmaps,axis=0)
            np.save(f'./data/{case}{scenario}/{fname}',heatmaps)
        self.interactive_heatmap_2d(heatmaps,method='music_aod')

    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """************** Reflection Coefficient 3D Imaging *********************"""
    """**********************************************************************"""
    """**********************************************************************"""
    """**********************************************************************"""
    def R_l(self, l,cur_loc):
        """
        l: Index of TxRxPair
        cur_loc: Targeting Location

        Purpose: compute the average distance between the target and the l-th pair of transmitter and receiver
        """
        tx_idx = self.txRxPairs[l,0]-1
        rx_idx = self.txRxPairs[l,1]-1
        x_m, y_m = self.ants_locations[tx_idx,:]
        x_n, y_n = self.ants_locations[rx_idx,:]
        z_m, z_n = 0.0, 0.0
        x_t, y_t, z_t = cur_loc
        return (np.sqrt((x_m-x_t)**2 + (y_m-y_t)**2 + (z_m-z_t)**2)+np.sqrt((x_n-x_t)**2 + (y_n-y_t)**2 + (z_n-z_t)**2)) / 2

    def S_t(self, S, l, Q, cur_loc):
        """
        S: Receieved Signal
        l: Index of TxRxPair
        Q: number of frequency points

        Purpose: Convert the receieved signal in to time domain with specified target location; for TDBP imaging method
        """

        cur_R = self.R_l(l, cur_loc)

        kq = 2*np.pi*self.freq/c
        shift_element = np.exp(1j*2*kq*cur_R,dtype='complex')
        
        St = S[l,:]*shift_element
        St = np.sum(St) / Q

        return St

    def reflection_coefficient(self, S, loc):
        print(f'current location: {loc}')
        MN = len(self.txRxPairs)
        corr_matrix = np.zeros((MN-1,MN-1),dtype='complex')
        Q = len(self.freq)
        St = []
        for i in range(MN):
            St.append(self.S_t(S,i,Q,loc))
        St = np.array(St)
        corr_matrix = St[:-1].reshape(1,-1) * St[1:].reshape(-1,1)
        return np.abs(np.sum(corr_matrix))
    
    def rc_3D_point_cloud(self, signal, x_grid=None, y_grid=None, z_grid=None):
        if x_grid is None:
            x_grid = np.arange(-2,2,0.05)
        if y_grid is None:
            y_grid = np.arange(-2,2,0.05)
        if z_grid is None:
            z_grid = np.arange(0,3,0.05)

        point_cloud = np.zeros((len(x_grid),len(y_grid),len(z_grid)))
        pool = mp.Pool()
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                for k, z in enumerate(z_grid):
                    output = pool.apply_async(self.reflection_coefficient,(signal, np.array([x,y,z])))
                    point_cloud[i,j,k] = output.get()
        pool.close()
        return point_cloud
    
    def plot_3D_point_cloud(self, point_cloud, meshgrids, th=0.8):
        print('plotting...')
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111,projection='3d')
        normalized_pc = self.normalization(point_cloud)
        target_idx = abs(normalized_pc)>th
        ax.scatter(meshgrids[0][target_idx], meshgrids[1][target_idx], meshgrids[2][target_idx])
        # x,y,z = np.nonzero(normalized_pc)
        # ax.scatter(x,y,z)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.grid(True)
        plt.show()

"""**********************************************************************"""
"""**********************************************************************"""
"""**********************************************************************"""
"""************************** Main Function *****************************"""
"""**********************************************************************"""
"""**********************************************************************"""
"""**********************************************************************"""
def main():
    plt.close()
    # current_case = 'test01312023/' # 2cf
    current_case = 'test01242023' # 1cf
    # current_case = 'test01162023/'
    my_vtrig = isens_vtrigU(case=current_case, peak_height=0.9, enhance_rate=100, enhance=True, interpolate=False)
    # my_vtrig.fft_pipeline(case=current_case, all_case=True, plot=None, test_mode=False, show_ants=False)
    # my_vtrig.fft_pipeline(case=current_case, scenario='human_y_angle_-20',cal_method=0 ,all_case=False, plot='both', test_mode=True, show_ants=False)
    # my_vtrig.range_pipeline(case=current_case, scenario='human_x_angle_0')
    # my_vtrig.music_pipeline(case=current_case, scenario='cf_x_angle_-20',signal_dimension=1)
    # my_vtrig.music_aoa_single_frame_pipeline(case=current_case, scenario='cf_y_angle_+20',signal_dimension=1, plot_aoa_spectrum=False)
    # my_vtrig.music_aod_single_frame_pipeline(case=current_case, scenario='cf_x_angle_0',signal_dimension=3, plot_aod_spectrum=False)
    # current_case = 'test01222023/' # x calibration
    # my_vtrig.music_aod_single_frame_pipeline(case=current_case, scenario='2cf_yy_angle_+-20',signal_dimension=5, plot_aod_spectrum=False)
    # angles = ['0','+20','+40']
    # for angle in angles:
    #     my_vtrig.music_pipeline(case=current_case, scenario=f'cf_y_angle_{angle}', mode='aod', simulation=False, cal_method=0, signal_dimension=1, smoothing=True, plot_spectrum=False)

    # current_case = 'test01162023/'
    # directions = ['x','y','z']
    # for direction in directions:
    #     my_vtrig.music_pipeline(case=current_case, scenario=f'cf_move_{direction}', mode='aod', simulation=False, cal_method=0, signal_dimension=1, smoothing=True, plot_spectrum=False)
    current_scenario = 'cf_y_angle_0'
    calArr, recArr = my_vtrig.load_data(case=current_case, scenario=current_scenario)
    proArr = my_vtrig.calibration(calArr,recArr,method=0)
    x_grid = np.arange(-2,2.5,0.5)
    y_grid = np.arange(-2,2.5,0.5)
    z_grid = np.arange( 0,3.05,0.05)
    x_grid_idx, y_grid_idx, z_grid_idx = np.meshgrid(x_grid, y_grid, z_grid)
    meshgrids = [x_grid_idx, y_grid_idx, z_grid_idx]
    point_cloud = my_vtrig.rc_3D_point_cloud(proArr[50,:,:],x_grid, y_grid, z_grid)
    my_vtrig.plot_3D_point_cloud(point_cloud, meshgrids, th=0.8)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:       
        print('')
    finally:
        print('')
        print("...Program terminated")