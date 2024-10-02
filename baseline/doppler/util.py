import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
from scipy.signal import find_peaks


def get_BPM(respiration_wave):
    peaks, _ = find_peaks(respiration_wave, width=3, height=0.5)
    index_time = np.array([i * 0.1 for i in range(len(respiration_wave))])
    time_intervals = np.diff(index_time[peaks])

    # Calculate the frequencies (in Hz)
    frequencies = 1 / time_intervals

    # Average frequency (rate in Hz)
    # Get average respiration rate
    if len(frequencies) == 0:
        average_frequency = 0.2
    else:
        average_frequency = np.mean(frequencies)
    # print(average_frequency)
    average_BPM = int(average_frequency * 60)
    return average_BPM


def plot_BPM(respiration_wave):
    """Plot the respiration wave and save it to a file."""
    plt.clf()
    plt.plot(respiration_wave)
    plt.savefig("temp.png")


class ProcessRadarSlidingWindow:
    num_tx = 10
    rbw = 50
    range_nfft = 400
    angle_nfft = [10, 20]

    def __init__(self, radar_window, radar_config):
        self.radar_window = radar_window
        # print(f"Radar window shape: {self.radar_window.shape}")
        self.radar_config = radar_config

        self.doppler_nfft = self.radar_window.shape[
            0
        ]  # @TODO: this could be the shape of the recording

    def process(self, method="aoa_aod", freq_range=(0.1, 2.0), dist_range=(0.3, 2.5)):
        # method: "aoa_aod" or "range_doppler"
        processed_data = self.radar_window - np.mean(
            self.radar_window[0:2, :, :], axis=0
        )
        self.freq_range = freq_range
        self.dist_range = dist_range

        if method == "range_doppler":
            range_profile = np.fft.ifft(processed_data, n=self.range_nfft, axis=2)
            range_profile_norm = np.linalg.norm(range_profile, axis=1)
            range_profile_norm[np.isnan(range_profile_norm)] = 0

            doppler = np.fft.fft(
                np.real(range_profile_norm), n=self.doppler_nfft, axis=0
            )
            doppler = doppler[0 : len(doppler) // 2, :]
            doppler = np.abs(doppler.T)
            doppler_slice, freq_range, dist_range = self.slice_range_doppler(
                doppler, freq_range=freq_range, dist_range=dist_range
            )
            # self.plot_range_doppler(doppler_slice, freq_range, dist_range)
            doppler_slice = (doppler_slice - np.min(doppler_slice)) / (
                np.max(doppler_slice) - np.min(doppler_slice)
            )

            return doppler_slice

        if method == "aoa_aod":
            # Now fft over axis 1,2, and 3 to get AoA, AoD, and range profile
            processed_data_3d = processed_data.reshape(
                processed_data.shape[0], self.num_tx, 20, processed_data.shape[-1]
            )
            processed_data_3d_range = np.fft.ifft(
                processed_data_3d, n=self.range_nfft, axis=3
            )
            processed_data_3d_range = np.fft.fft2(
                processed_data_3d_range, s=self.angle_nfft, axes=(1, 2)
            )

            dopplers = []

            for i in range(self.angle_nfft[0]):
                for j in range(self.angle_nfft[1]):
                    value = processed_data_3d_range[:, i, j, :]
                    doppler = np.fft.fft(np.real(value), n=self.doppler_nfft, axis=0)
                    doppler_slice = doppler[0 : len(doppler) // 2, :]
                    doppler_slice = np.abs(doppler_slice.T)
                    # print(f"Shape of doppler slice before: {doppler_slice.shape}")

                    doppler_slice, freq_range, dist_range = self.slice_range_doppler(
                        doppler_slice, freq_range=(0.1, 2.0), dist_range=(0.2, 2.5)
                    )
                    # print(doppler_slice.shape)
                    self.plot_result(doppler_slice, freq_range, dist_range)
                    input("Press any key to continue...")

                    # Normalize the doppler slice between 0 and 1
                    doppler_slice = (doppler_slice - np.min(doppler_slice)) / (
                        np.max(doppler_slice) - np.min(doppler_slice)
                    )
                    dopplers.append(doppler_slice)

            dopplers = np.array(dopplers)
            return dopplers

        raise ValueError("Invalid method")

    def slice_range_doppler(
        self, range_doppler, freq_range=(0.1, 2.0), dist_range=(0.5, 2.5)
    ):
        freq = self.radar_config["freq"]
        Ts = 1 / self.range_nfft / (freq[1] - freq[0] + 1e-16)  # Avoid nan checks
        time_vec = np.linspace(0, Ts * (self.range_nfft - 1), num=self.range_nfft)
        dist_vec = time_vec * (c / 2)  # distance in meters
        self.dist_vec_step = dist_vec[1] - dist_vec[0]

        d = self.radar_config["sample_time"]
        doppler_freq = np.fft.fftfreq(self.doppler_nfft, d)
        doppler_freq = doppler_freq[doppler_freq >= 0]
        self.doppler_freq_step = doppler_freq[1] - doppler_freq[0]

        range_low_idx = np.where(dist_vec >= dist_range[0])[0][0]
        range_high_idx = np.where(dist_vec <= dist_range[1])[0][-1]
        freq_low_idx = np.where(doppler_freq >= freq_range[0])[0][0]
        freq_high_idx = np.where(doppler_freq <= freq_range[1])[0][-1]

        freq_low, freq_high = doppler_freq[freq_low_idx], doppler_freq[freq_high_idx]
        range_low, range_high = dist_vec[range_low_idx], dist_vec[range_high_idx]
        doppler_slice = range_doppler[
            range_low_idx:range_high_idx, freq_low_idx:freq_high_idx
        ]
        return doppler_slice, (freq_low, freq_high), (range_low, range_high)

    def plot_doppler_slice(self, doppler_slice, freq_range, dist_range):
        extent = [freq_range[0], freq_range[1], dist_range[0], dist_range[1]]
        plt.figure(figsize=(8, 6))
        plt.imshow(doppler_slice, origin="lower", extent=extent, aspect="auto")
        # plt.legend()
        plt.colorbar()
        plt.xlabel("Doppler Frequency [Hz]")
        plt.ylabel("Range [m]")
        plt.title(f"Range-Doppler Vital Sign Heatmap: {10} Tx {50} [Hz]")
        plt.grid()
        plt.savefig("heatmap.png")

    def plot_max_freq(self, doppler_slice, max_freq, max_freq_dist):
        extent = [
            self.freq_range[0],
            self.freq_range[1],
            self.dist_range[0],
            self.dist_range[1],
        ]
        plt.figure(figsize=(8, 6))
        plt.imshow(doppler_slice, origin="lower", extent=extent, aspect="auto")
        # plt.legend()
        plt.colorbar()
        plt.xlabel("Doppler Frequency [Hz]")
        plt.ylabel("Range [m]")
        plt.title(f"Range-Doppler Vital Sign Heatmap: {10} Tx {50} [Hz]")
        plt.grid()

        plt.scatter(max_freq, max_freq_dist, c="r", s=100, marker="x")
        # plt.show()
        plt.savefig("bsense_baseline_doppler_result.png")

    def get_max_freq(self, doppler_slice):
        dist_idx, freq_idx = np.unravel_index(
            doppler_slice.argmax(), doppler_slice.shape
        )
        # print("Max value: ", doppler_slice[max_idx])
        freq = (
            self.freq_range[0]
            + freq_idx * self.doppler_freq_step
            + self.doppler_freq_step / 2
        )
        dist = (
            self.dist_range[0] + dist_idx * self.dist_vec_step + self.dist_vec_step / 2
        )
        # print("Doppler freq: ", freq)
        return freq, dist
