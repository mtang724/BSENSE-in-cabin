from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.constants import c


class BaseCase:
    num_tx = 10
    rbw = 50
    range_nfft = 512
    angle_nfft = [6, 6]

    def __init__(self, case_path):
        self.case_path = case_path
        self.recording = None
        self.config = None
        self.respiration_gt = None
        self.range_profile = None

        self.read_case()
    
    def read_case(self):
        self.recording = np.load(os.path.join(self.case_path, 'recording.npy'))
        self.config = np.load(os.path.join(self.case_path, 'config.npy'), allow_pickle=True).item()
        self.respiration_gt = np.load(os.path.join(self.case_path, 'respiration_gt.npy'))


class CaseProcessor(BaseCase):
    
    def run(self, plot=False):
        # First create sliding window for the recording and match with respiration_gt
        self.range_profile, self.dist_vec = self.get_range_profile(self.config)
        self.dist_vec_step = self.get_dist_vec_step(self.dist_vec)

        # Sliding window on frames for range-doppler
        window_size = 100
        step_size = 10
        max_freq_list, max_freq_dist_list = [], []
        # print("Total frames: ", self.recording.shape[0])
        for i in range(0, self.recording.shape[0]-window_size, step_size):
            frame_range = (i, i+window_size)
            max_freq, max_freq_dist = self.process_window(frame_range, plot=plot)
            max_freq_list.append(max_freq)
            max_freq_dist_list.append(max_freq_dist)

        # Get the index range for each max_freq_dist
        index = [(i, i+window_size) for i in range(0, self.recording.shape[0]-window_size, step_size)]

        return max_freq_list, max_freq_dist_list, index

    def process_window(self, frame_range, plot=False):
        # print("Processing frame range: ", frame_range)
        doppler, doppler_freq = self.get_range_doppler(frame_range)
        self.doppler_freq_step = self.get_doppler_freq_step(doppler_freq)

        doppler_slice, freq_range, dist_range = self.slice_range_doppler(doppler, doppler_freq, dist_range=(0.5, 1.5))
        # print("freq range: ", freq_range, "dist range: ", dist_range)
        max_freq, max_freq_dist = self.get_max_freq(doppler_slice, freq_range, dist_range)
        # print("max_freq: ", max_freq, "max_freq_dist: ", max_freq_dist)

        if plot:
            self.plot_result(doppler_slice, max_freq, max_freq_dist, freq_range, dist_range)

        return max_freq, max_freq_dist

    def plot_result(self, doppler_slice, max_freq, max_freq_dist, freq_range, dist_range):
        extent = [freq_range[0], freq_range[1], dist_range[0], dist_range[1]]
        plt.figure(figsize=(8,6))
        plt.imshow(doppler_slice, origin='lower', extent=extent, aspect='auto')
        # plt.legend()
        plt.colorbar()
        plt.xlabel("Doppler Frequency [Hz]")
        plt.ylabel("Range [m]")
        plt.title(f"Range-Doppler Vital Sign Heatmap: {10} Tx {50} [Hz]")
        plt.grid()

        plt.scatter(max_freq, max_freq_dist, c='r', s=100, marker='x')
        plt.show()

    def get_range_profile(self, config):
        processed_data = self.recording - np.mean(self.recording[0:3, :, :], axis=0)
        range_profile = np.fft.ifft(processed_data, n=self.range_nfft, axis=2)

        freq = config['freq']
        Ts = 1/self.range_nfft/(freq[1]-freq[0]+1e-16) # Avoid nan checks
        time_vec = np.linspace(0,Ts*(self.range_nfft-1),num=self.range_nfft)
        dist_vec = time_vec*(c/2) # distance in meters

        range_profile_norm = np.linalg.norm(range_profile, axis=1)
        range_profile_norm[np.isnan(range_profile_norm)] = 0

        return range_profile_norm, dist_vec
    
    def get_dist_vec_step(self, dist_vec):
        return dist_vec[1] - dist_vec[0]
    
    def get_range_doppler(self, frame_range=(0, 100)):
        # Range-Doppler Profile
        n=frame_range[1]-frame_range[0]
        doppler = np.fft.fft(np.real(self.range_profile[frame_range[0]:frame_range[1]]), n=n,  axis=0)
        doppler = doppler[0:len(doppler)//2, :]
        doppler = np.abs(doppler.T)

        d = self.config["sample_time"]
        # print("d", d)
        doppler_freq = np.fft.fftfreq(n, d)
        doppler_freq = doppler_freq[doppler_freq>=0]

        return doppler, doppler_freq
    
    def get_doppler_freq_step(self, doppler_freq):
        return doppler_freq[1] - doppler_freq[0]
    
    def slice_range_doppler(self, range_doppler, doppler_freq, freq_range=(0.1, 2.0), dist_range=(0.5, 1.5)):
        range_low_idx = np.where(self.dist_vec>=dist_range[0])[0][0]
        range_high_idx = np.where(self.dist_vec<=dist_range[1])[0][-1]
        freq_low_idx = np.where(doppler_freq>=freq_range[0])[0][0]
        freq_high_idx = np.where(doppler_freq<=freq_range[1])[0][-1]

        freq_low, freq_high = doppler_freq[freq_low_idx], doppler_freq[freq_high_idx]
        range_low, range_high = self.dist_vec[range_low_idx], self.dist_vec[range_high_idx]
        doppler_slice = range_doppler[range_low_idx:range_high_idx, freq_low_idx:freq_high_idx]
        return doppler_slice, (freq_low, freq_high), (range_low, range_high)

    def get_max_freq(self, doppler_slice, freq_range, dist_range):
        dist_idx, freq_idx = np.unravel_index(doppler_slice.argmax(), doppler_slice.shape)
        # print("Max value: ", doppler_slice[max_idx])
        freq = freq_range[0] + freq_idx * self.doppler_freq_step + self.doppler_freq_step / 2
        dist = dist_range[0] + dist_idx * self.dist_vec_step + self.dist_vec_step / 2
        # print("Doppler freq: ", freq)
        return freq, dist

if __name__ == "__main__":
    data_path = "0131Aligned"
    case_path = os.path.join(data_path, "50k_profile_1_distance0.5_degree_0_round_2_Hanbo")
    case = CaseProcessor(case_path)
    max_freq_list, max_freq_dist_list, index = case.run(plot=True)
    print(max_freq_list)
    print(index[0])
    
    # doppler, doppler_freq, doppler_freq_step = case.get_range_doppler()
    # print(doppler_freq_step)
