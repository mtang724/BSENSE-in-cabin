import math
import numpy as np
from scipy.constants import c
from scipy.signal import butter, filtfilt, find_peaks
from scipy import fftpack
import scipy.signal as signal
from scipy.ndimage import binary_closing, binary_opening

# Calculate the antenna spacing for Vayyar V_Trig Radar
def distance_3d(x1, y1, z1, x2, y2, z2):
    """
    Calculate the Euclidean distance between two points in 3D space.

    Parameters:
    x1 (float): The x-coordinate of the first point.
    y1 (float): The y-coordinate of the first point.
    z1 (float): The z-coordinate of the first point.
    x2 (float): The x-coordinate of the second point.
    y2 (float): The y-coordinate of the second point.
    z2 (float): The z-coordinate of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def MVDR_beamforming(range_profile, num_tx = 20, num_rx = 20, searchStep = 10):
    """
    Perform Minimum Variance Distortionless Response (MVDR) beamforming on the given range profile data.
    Parameters:
    range_profile (numpy.ndarray): The input range profile data with dimensions (frames, num_rx, range_index).
    num_tx (int, optional): Number of transmit antennas. Default is 20.
    num_rx (int, optional): Number of receive antennas. Default is 20.
    searchStep (int, optional): Step size for the azimuth and elevation field of view search. Default is 10.
    Returns:
    tuple: A tuple containing:
        - rangeAngle (numpy.ndarray): The range-angle profile.
        - bfWeight (numpy.ndarray): The beamforming weights.
        - invCovMat (numpy.ndarray): The pseudoinverse of the covariance matrices.
        - beamformed_signal_frames_range (numpy.ndarray): The beamformed signal frames in the range domain.
    """
    # tx_spacing = distance_3d(-0.0275, -0.0267, 0, -0.0253, -0.0267, 0)
    rx_spacing = distance_3d(0.0274, -0.0133, 0, 0.0274, -0.0112, 0)
    # fs = 64e9
    fs = 62e9
    wavelength = 3e8/fs # Wavelength of the radar signal
    d_over_lambda = rx_spacing / wavelength
    
    # Azimuth, Elevation, fov search
    afov = 160
    efov = 160
    # searchStep = 10

    # Calculate angular ranges
    efov_rad = np.radians(np.linspace(-efov/2, efov/2, int(efov/searchStep)))
    afov_rad = np.radians(np.linspace(-afov/2, afov/2, int(afov/searchStep)))

    # Sine values for the angles
    a1 = np.sin(afov_rad)
    a2 = np.sin(efov_rad)

    # Initialize the steering matrix A
    L = num_tx * num_rx
    A = np.zeros((L, len(a1) * len(a2)), dtype=complex)

    # Grid generation for element indices
    m_ind = - np.array([value for value in range(num_tx) for _ in range(num_rx)]) # Azimuth antenna indexes
    n_ind = - np.array([i for i in range(num_rx)] * num_tx) # Elevation antenna indexes

    # Fill in the steering matrix
    for ii in range(len(a2)):
        for jj in range(len(a1)):
            phase_shift = -2j * np.pi * d_over_lambda * (m_ind * a1[jj] + n_ind * a2[ii])
            A[:, ii * len(a1) + jj] = np.exp(phase_shift)
    steerMat = A
    
    # frames, nAnt, range_index
    # range_profile = np.transpose(range_profile, (1, 0, 2))
    
    M = range_profile.shape[2]  # Number of range bins

    # Initialize outputs
    condNumber = np.zeros(M)
    rangeAngle = np.zeros((M, steerMat.shape[1]))  # DOA angles
    bfWeight = np.zeros((M, steerMat.shape[0], steerMat.shape[1]), dtype="complex")
    invCovMat = np.zeros((M, num_tx * num_rx, num_tx * num_rx), dtype="complex")  # Adjust size as needed
    beamformed_signal_frames_range = np.zeros((M, steerMat.shape[1], range_profile.shape[1]), dtype="complex")

    for rIdx in range(M):  # Iterate over range bins
        # Extract signal matrix for the current range bin
        sigMat = range_profile[:, :, rIdx].squeeze()  # Adjust slicing as needed
        # Estimate covariance matrix
        covMat = np.dot(sigMat, sigMat.conj().T) / sigMat.shape[1]  # Normalize by the actual number of samples
        # Calculate condition number and apply diagonal loading
        condNumber[rIdx] = np.linalg.cond(covMat)
        alpha = 0.003 * np.mean(np.diag(covMat))  # Adjust alpha as needed
        covMat += alpha * np.eye(covMat.shape[0])
        # Capon beamforming (MVDR)
        pinv_covMat = np.linalg.pinv(covMat)
        bfWeight_ = np.dot(pinv_covMat, steerMat)
        # Calculate range-angle profile
        rangeAngle[rIdx, :] = 1 / np.abs(np.diag(np.dot(steerMat.conj().T, bfWeight_)))
        # Store the pseudoinverse of the covariance matrix
        invCovMat[rIdx, :, :] = pinv_covMat
        bfWeight[rIdx, :, :] = bfWeight_
        beamformed_signal_frames_range[rIdx, :] = np.dot(bfWeight_.conj().T, sigMat)
        
    return rangeAngle, bfWeight, invCovMat, beamformed_signal_frames_range


def MVDR_beamforming_with_phase(range_profile, num_tx=10, num_rx=20, searchStep=10):
    """
    Perform MVDR beamforming with phase information extraction.

    Parameters:
    range_profile (numpy.ndarray): The range profile data with dimensions [num_antennas, num_ranges, num_frames].
    num_tx (int, optional): Number of transmit antennas. Default is 10.
    num_rx (int, optional): Number of receive antennas. Default is 20.
    searchStep (int, optional): Step size for the search grid in degrees. Default is 10.

    Returns:
    tuple: A tuple containing:
        - rangeAngle (numpy.ndarray): The range-angle map with dimensions [num_ranges, num_directions].
        - bfWeight (numpy.ndarray): The beamforming weights with dimensions [num_ranges, num_antennas, num_directions].
        - invCovMat (numpy.ndarray): The inverse covariance matrices with dimensions [num_ranges, num_antennas, num_antennas].
        - phaseInfo (numpy.ndarray): The phase information with dimensions [num_frames, num_ranges, num_directions].
    """
    
    rx_spacing = distance_3d(0.0274, -0.0133, 0, 0.0274, -0.0112, 0)
    fc = 62e9  # Central frequency of the radar signal
    wavelength = 3e8 / fc
    d_over_lambda = rx_spacing / wavelength

    afov = 160
    efov = 160

    efov_rad = np.radians(np.linspace(-efov / 2, efov / 2, int(efov / searchStep)))
    afov_rad = np.radians(np.linspace(-afov / 2, afov / 2, int(afov / searchStep)))

    a1 = np.sin(afov_rad)
    a2 = np.sin(efov_rad)

    L = num_tx * num_rx
    A = np.zeros((L, len(a1) * len(a2)), dtype=complex)

    m_ind = - np.array([value for value in range(num_tx) for _ in range(num_rx)])
    n_ind = - np.array([i for i in range(num_rx)] * num_tx)

    for ii in range(len(a2)):
        for jj in range(len(a1)):
            phase_shift = -2j * np.pi * d_over_lambda * (m_ind * a1[jj] + n_ind * a2[ii])
            A[:, ii * len(a1) + jj] = np.exp(phase_shift)
    steerMat = A

    num_antennas = range_profile.shape[0]
    num_ranges = range_profile.shape[1]
    num_frames = range_profile.shape[2]

    condNumber = np.zeros(num_ranges)
    rangeAngle = np.zeros((num_ranges, steerMat.shape[1]))
    bfWeight = np.zeros((num_ranges, steerMat.shape[0], steerMat.shape[1]), dtype=complex)
    invCovMat = np.zeros((num_ranges, num_antennas, num_antennas), dtype=complex)
    phaseInfo = np.zeros((num_frames, num_ranges, steerMat.shape[1]))  # Dimensions: [time, range, direction]

    for rIdx in range(num_ranges):
        sigMat = range_profile[:, rIdx, :].squeeze().T  # Transpose to get [frames, antennas]
        covMat = np.dot(sigMat.T, sigMat.conj()) / sigMat.shape[0]
        condNumber[rIdx] = np.linalg.cond(covMat)
        alpha = 0.003 * np.mean(np.diag(covMat))
        covMat += alpha * np.eye(covMat.shape[0])
        pinv_covMat = np.linalg.pinv(covMat)
        bfWeight_ = np.dot(pinv_covMat, steerMat)
        rangeAngle[rIdx, :] = 1 / np.abs(np.diag(np.dot(steerMat.conj().T, bfWeight_)))
        invCovMat[rIdx, :, :] = pinv_covMat
        bfWeight[rIdx, :, :] = bfWeight_

        for dir_idx in range(steerMat.shape[1]):
            beamformed_signal = np.dot(sigMat, bfWeight_[:, dir_idx])
            if beamformed_signal.shape[0] != num_frames:
                raise ValueError(f"Mismatch in beamformed signal length: expected {num_frames}, got {beamformed_signal.shape[0]}")
            phaseInfo[:, rIdx, dir_idx] = np.unwrap(np.angle(beamformed_signal))

    return rangeAngle, bfWeight, invCovMat, phaseInfo

def calculate_microdoppler(phaseInfo, fs, nperseg=256, noverlap=128):
    num_frames, num_range_bins, num_directions = phaseInfo.shape
    microDoppler = np.zeros((num_frames, num_range_bins, num_directions))

    for rIdx in range(num_range_bins):
        for dir_idx in range(num_directions):
            # unwrapped_phase = np.unwrap(phaseInfo[:, rIdx, dir_idx])
            unwrapped_phase = phaseInfo[:, rIdx, dir_idx]
            nperseg_adjusted = min(nperseg, len(unwrapped_phase))
            f, t, Sxx = signal.spectrogram(unwrapped_phase, fs, nperseg=nperseg_adjusted, noverlap=noverlap)
            microDoppler[:, rIdx, dir_idx] = np.mean(Sxx, axis=0)
    
    return f, t, microDoppler

def extract_beamformed_signal_per_range(sigMat, bfWeight, rIdx_low, rIdx_range):
    for rIdx in range(rIdx_low, rIdx_range):
        weights_angle = bfWeight[rIdx, :, :]
        

def extract_beamformed_signal_frames(sigMat, bfWeight, rIdx, aIdx):
    """
    Extracts beamformed signal frames from the given signal matrix using the provided beamforming weights.
    Args:
        sigMat (np.ndarray): A 3D numpy array of complex signals with dimensions (num_antennas, num_frames, num_range_bins).
        bfWeight (np.ndarray): A 3D numpy array of complex beamforming weights with dimensions (num_range_bins, num_antennas, num_angles).
        rIdx (int): The index of the range bin to process.
        aIdx (int): The index of the angle to process.
    Returns:
        np.ndarray: A 1D numpy array of complex beamformed signal frames with length equal to the number of frames.
    Raises:
        ValueError: If the weights array or the signal array at the specified frame and range bin is not 1D.
    """
    num_frames = sigMat.shape[1]
    beamformed_signal_frames = np.zeros(num_frames, dtype=np.complex64)
    
    for frame_idx in range(num_frames):
        # Assuming bfWeight[rIdx, aIdx] correctly indexes into a 1D array of weights
        weights = bfWeight[rIdx, :, aIdx]
        
        if weights.ndim != 1:
            raise ValueError("Weights array is not 1D.")
        
        # Extract signal for all antennas at the current frame and range bin
        signal_at_frame_and_range = sigMat[:, frame_idx, rIdx]
        
        if signal_at_frame_and_range.ndim != 1:
            raise ValueError("Signal array is not 1D.")
        
        # Perform dot product
        beamformed_signal_frames[frame_idx] = np.dot(weights.conj().T, signal_at_frame_and_range)

    return beamformed_signal_frames
        
        
def CFAR1D_CASO(signal, K0):
    """
    Perform 1D Constant False Alarm Rate (CFAR) detection using Cell Averaging Smallest Of (CASO) method.
    Parameters:
    signal (numpy.ndarray): 2D array representing the range-angle data from beamforming.
    K0 (float): CFAR threshold scaling factor.
    Returns:
    tuple: A tuple containing:
        - Detect (dict): A dictionary with detection results:
            - 'nObject' (int): Number of detected objects.
            - 'index' (list): List of indices of detected objects.
            - 'cut_power' (list): List of power values at the detected object locations.
            - 'noise_power' (list): List of noise power estimates at the detected object locations.
            - 'threshold' (list): List of threshold values used for detection.
        - detSnrProfile (numpy.ndarray): 2D array representing the detection SNR profile.
    """
    # signal - rangeAngle from Beamforming
    refWinSize         =   [8, 8]
    guardWinSize       =   [6, 4]         # avoid the first elevation harmonic
    # K0                 =   [6, 6]    # original parameter
    
    cellNum = refWinSize[0]
    gapNum = guardWinSize[0]
    K0 = K0



    N_rng, N_azi = signal.shape

    gaptot = gapNum + cellNum
    N_obj = 0
    Ind_obj = []
    noise_obj = []

    discardRngCellLeft =   5 # 12;  
    discardRngCellRight=   2 #32 % original parameter is 32; %8; %%Anand changed from 32 to 2 on March 25, 2021 for baby in van
    discardCellLeft    =   1
    discardCellRight   =   1

    # Initialize the detection matrix
    detSnrProfile = np.zeros((N_rng, N_azi))

    # Loop over azimuth bins
    for k in range(discardCellLeft, N_azi - discardCellRight):
        sigv = signal[:, k]

        # Prepare the signal vector with padding
        meanLeft = np.mean(sigv[discardRngCellLeft:2 + discardRngCellLeft])
        meanRight = np.mean(sigv[-2 - discardRngCellRight:-discardRngCellRight])
        vecLeft = np.full(gaptot + discardRngCellLeft, meanLeft)
        vecRight = np.full(gaptot + discardRngCellRight, meanRight)
        vec = np.concatenate([vecLeft, sigv[discardRngCellLeft:N_rng - discardRngCellRight], vecRight])

        # CFAR detection
        for j in range(N_rng):
            cellInda = np.arange(j - gaptot, j - gapNum) + gaptot
            cellIndb = np.arange(j + gapNum + 1, j + gaptot + 1) + gaptot
            cellave1a = np.sum(vec[cellInda]) / cellNum
            cellave1b = np.sum(vec[cellIndb]) / cellNum

            # Additional noise estimation areas
            cellIndc = np.arange(j - 5, j - 1) + gaptot
            cellIndd = np.arange(j + 2, j + 6) + gaptot
            cellave1c = np.mean(vec[cellIndc])
            cellave1d = np.mean(vec[cellIndd])

            cellave1 = min([cellave1a, cellave1b, cellave1c, cellave1d])

            detSnrProfile[j, k] = vec[j + gaptot] / (K0 * cellave1)

            if discardRngCellLeft < j < (N_rng - discardRngCellRight):
                if vec[j + gaptot] > K0 * cellave1:
                    N_obj += 1
                    Ind_obj.append([j, k])
                    noise_obj.append(cellave1)

    # Prepare the detection structure
    Detect = {
        'nObject': N_obj,
        'index': Ind_obj,  # Limiting to maxObjects as in MATLAB code
        'cut_power': [vec[j + gaptot] for j, _ in Ind_obj],
        'noise_power': noise_obj,
        'threshold': [K0] * min(N_obj, 512)
    }

    return Detect, detSnrProfile

def find_range_angle_Idx(rangeAngle):
    """
    Finds the indices of the maximum value in a 2D array after summing over one axis.

    Args:
        rangeAngle (np.ndarray): A 2D numpy array where the first axis represents range and the second axis represents angle.

    Returns:
        tuple: A tuple (rIdx, aIdx) where rIdx is the index of the maximum value along the range axis after summing over angles,
               and aIdx is the index of the maximum value along the angle axis for the identified range index.
    """
    cut_off_range_angle = rangeAngle
    sum_over_angles = np.sum(cut_off_range_angle, axis=1)
    # Find the index of the maximum value
    max_value_index = np.argmax(sum_over_angles)
    aIdx = np.argmax(cut_off_range_angle[max_value_index])
    rIdx = max_value_index
    return rIdx, aIdx


def preprocess_signal(signal, fs):
    """
    Preprocesses a given signal by detrending and applying a bandpass filter.

    Parameters:
    signal (array-like): The input signal to be processed.
    fs (float): The sampling frequency of the signal.

    Returns:
    array-like: The processed signal after detrending and bandpass filtering.
    """
    # Detrend and bandpass filter the signal
    detrended_signal = signal - np.mean(signal)
    lowcut = 0.1
    highcut = 0.7
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, detrended_signal)
    return filtered_signal

def extract_respiration_rate(signal, fs):
    """
    Extracts the respiration rate from a given signal using Fourier Transform.
    Parameters:
    signal (numpy.ndarray): The input signal from which to extract the respiration rate.
    fs (float): The sampling frequency of the input signal.
    Returns:
    float: The estimated respiration rate in breaths per minute (BPM).
    Notes:
    The function focuses on the frequency range typically associated with human respiration (0.1 Hz to 0.7 Hz).
    """
    # Fourier Transform to identify frequency components
    fft_output = fftpack.fft(signal)
    fft_freq = fftpack.fftfreq(len(signal), d=1/fs)
    
    # Focus on the respiration frequency range
    valid_idx = np.where((fft_freq >= 0.1) & (fft_freq <= 0.7))
    fft_output_filtered = fft_output[valid_idx]
    fft_freq_filtered = fft_freq[valid_idx]
    
    # Identify the dominant frequency
    dominant_freq_idx = np.argmax(np.abs(fft_output_filtered))
    dominant_respiration_freq = fft_freq_filtered[dominant_freq_idx]
    
    # Convert to breaths per minute
    respiration_rate_bpm = dominant_respiration_freq * 60
    return respiration_rate_bpm


def compensate_phase_and_sum(signals, phase_shifts):
    """
    Compensate for phase differences in signals across range bins and sum them coherently for each frame.

    Parameters:
    - signals: numpy array of complex numbers with shape [num_range_bins, num_frames], 
               representing the signals from different range bins across frames.
    - phase_shifts: numpy array of phase shifts (in radians) to apply to each range bin, shape [num_range_bins].

    Returns:
    - summed_signals_per_frame: 1D numpy array of complex numbers, representing the coherently summed signal for each frame.
    """
    phase_shifts = phase_shifts.astype(signals.dtype)
    # Initialize an array to hold the summed signal for each frame
    summed_signals_per_frame = np.zeros(signals.shape[2], dtype='complex')
    
    # Compensate for the phase shifts with the correct sign across all frames for each range bin
    for i in range(signals.shape[0]):  # Iterate over range bins
        signals[i, :, :] *= np.exp(-1j * phase_shifts[i])  # Apply phase shift for this range bin across all frames
    
    # Sum the signals coherently across range bins for each frame
    summed_signals_per_frame = np.sum(signals, axis=0)
    
    return summed_signals_per_frame


def calculate_phase_shifts(frequency, tofs):
    """
    Calculate phase shifts for given time of flights (tofs) at a specific frequency.

    Parameters:
    - frequency: Signal frequency in Hz.
    - tofs: Array of time of flight values for each range bin, in seconds.

    Returns:
    - phase_shifts: Array of phase shifts in radians for each range bin.
    """
    c = 3e8  # Speed of light in m/s
    wavelength = c / frequency  # Calculate the wavelength
    phase_shifts = np.zeros(tofs.shape[0], dtype='complex')
    for i in range(tofs.shape[0]):
        phase_shifts[i] = 2 * np.pi * (tofs[i] % wavelength) / wavelength  # Calculate phase shifts, modulo the wavelength to handle cycles
    return phase_shifts


def aggregate_fft_bins(signal, fs, bin_ranges):
    """
    Perform FFT on a signal and aggregate the results into specified frequency bins.
    
    Parameters:
    - signal: The input signal.
    - fs: The sampling frequency.
    - bin_ranges: A list of tuples indicating the frequency ranges (start, stop) for aggregation.
    
    Returns:
    - A dictionary where keys are the frequency ranges and values are the aggregated FFT magnitudes.
    """
    
    # Perform FFT
    n = len(signal)
    freq_spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=1/fs)
    
    # Frequency resolution
    freq_res = fs / n
    
    # Prepare output dictionary
    aggregated_bins = {}
    
    # Aggregate FFT bins
    for start, stop in bin_ranges:
        # Identify bins within the range
        bin_indices = np.where((freqs >= start) & (freqs < stop))[0]
        
        # Aggregate magnitude (using absolute value of FFT result)
        aggregated_magnitude = np.sum(np.abs(freq_spectrum[bin_indices]))
        
        # Store in dictionary
        aggregated_bins[f"{start}-{stop}Hz"] = aggregated_magnitude
        
    return aggregated_bins

def process_signal(signal, fs, bin_ranges):
    """
    Process a signal with dimensions [num_instance, angles, frames] to aggregate FFT bins.
    
    Parameters:
    - signal: The input signal with shape [num_instance, angles, frames].
    - fs: The sampling frequency.
    - bin_ranges: A list of tuples indicating the frequency ranges (start, stop) for aggregation.
    
    Returns:
    - A numpy array with shape [num_instance, angles, len(bin_ranges)] containing aggregated FFT magnitudes.
    """
    num_instance, angles, frames = signal.shape
    aggregated_bins = np.zeros((num_instance, angles, len(bin_ranges)))
    
    for i in range(num_instance):
        for j in range(angles):
            # Extract the signal slice for this instance and angle
            signal_slice = signal[i, j, :]
            
            # Use the aggregate_fft_bins function on this slice
            bin_results = aggregate_fft_bins(signal_slice, fs, bin_ranges)
            
            # Fill in the results for this instance and angle
            for k, bin_range in enumerate(bin_ranges):
                key = f"{bin_range[0]}-{bin_range[1]}Hz"
                aggregated_bins[i, j, k] = bin_results[key]
    
    return aggregated_bins


def process_signal_fft(signal, config):
    """
    Process a signal with dimensions [num_instance, angles, frames] to aggregate FFT bins.
    
    Parameters:
    - signal: The input signal with shape [num_instance, angles, frames].
    - fs: The sampling frequency.
    - bin_ranges: A list of tuples indicating the frequency ranges (start, stop) for aggregation.
    
    Returns:
    - A numpy array with shape [num_instance, angles, len(bin_ranges)] containing aggregated FFT magnitudes.
    """
    d = config["sample_time"]
    num_instance, angles, frames = signal.shape
    doppler_freq = np.fft.fftfreq(signal.shape[-1], d)
    doppler_freq = doppler_freq[doppler_freq >= 0]
    freq_low = np.where(doppler_freq >= 0.2)[0][0]
    freq_high = np.where(doppler_freq <= 2)[0][-1]
    aggregated_bins = np.zeros((num_instance, angles, freq_high - freq_low))

    for i in range(num_instance):
        for j in range(angles):
            # Extract the signal slice for this instance and angle
            signal_slice = signal[i, j, :]

            # Generate Hanning window
            hanning_window = np.hanning(len(signal_slice))
            # Apply Hanning window to the signal slice
            windowed_signal_slice = signal_slice * hanning_window

            # Perform FFT on the windowed signal slice
            rd_map = np.fft.fft(np.real(windowed_signal_slice), n=signal.shape[-1] * 2, axis=-1)
            rd_map = rd_map[:len(rd_map) // 2]
            rd_map = np.abs(rd_map)

            # Populate the aggregated bins
            aggregated_bins[i, j] = rd_map[freq_low:freq_high]
    # range_low = np.where(dist_vec>=0.4)[0][0]
    # range_high = np.where(dist_vec<=2)[0][-1]
    # print(freq_low, freq_high,range_low, range_high)
    # rd_map = rd_map[20:330, 10:55]
    # rd_map = rd_map[:, freq_low:freq_high]
    
    return aggregated_bins

def aggregate_fft(signal):
    """
    Perform FFT on a signal and aggregate the results into specified frequency bins.
    
    Parameters:
    - signal: The input signal.
    - fs: The sampling frequency.
    - bin_ranges: A list of tuples indicating the frequency ranges (start, stop) for aggregation.
    
    Returns:
    - A dictionary where keys are the frequency ranges and values are the aggregated FFT magnitudes.
    """
    
    # Perform FFT
    n = signal.shape[-1]
    freq_spectrum = np.fft.fft(np.real(signal), n=n, axis=-1)
    return freq_spectrum

def preprocess_signal(signal, fs):
    # Apply bandpass filtering
    
    lowcut = 0.1  # Lower frequency bound (in Hz) for respiration rate
    highcut = 4.0  # Upper frequency bound (in Hz) for heart rate
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def range_doppler_processing(signal, N_fft_range, N_fft_doppler, config):
    """
    Convert time-domain radar signal to a Range-Doppler map.
    
    Parameters:
    - signal: The time-domain radar signal. Expected shape is (time_samples,).
    - N_fft_range: FFT size for range dimension.
    - N_fft_doppler: FFT size for doppler dimension.

    Returns:
    - rd_map: The Range-Doppler map.
    """
    range_nfft = N_fft_range
    freq = config['freq']
    Ts = 1/range_nfft/(freq[1]-freq[0]+1e-16) # Avoid nan checks
    time_vec = np.linspace(0,Ts*(range_nfft-1),num=range_nfft)
    dist_vec = time_vec*(c/2) # distance in meters
    
    rd_map = np.fft.fft(np.real(signal), n=N_fft_doppler, axis=0)
    rd_map = rd_map[0:len(rd_map)//2,:]
    rd_map = np.abs(rd_map.T)
    d = config["sample_time"]
    doppler_freq = np.fft.fftfreq(N_fft_doppler, d)
    doppler_freq = doppler_freq[doppler_freq>=0]

    freq_low = np.where(doppler_freq>=0.1)[0][0]
    freq_high = np.where(doppler_freq<=2)[0][-1]
    rd_map = rd_map[:, freq_low:freq_high]
    return np.abs(rd_map)