import math
import numpy as np
from scipy.constants import c

# Calculate the antenna spacing for Vayyar V_Trig Radar
def distance_3d(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def MVDR_beamforming(range_profile, num_tx = 20, num_rx = 20):
    # tx_spacing = distance_3d(-0.0275, -0.0267, 0, -0.0253, -0.0267, 0)
    rx_spacing = distance_3d(0.0274, -0.0133, 0, 0.0274, -0.0112, 0)

    wavelength = 3e8/64e9 # Wavelength of the radar signal
    d_over_lambda = rx_spacing / wavelength
    
    # Azimuth, Elevation, fov search
    afov = 180
    efov = 180
    searchStep = 10

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
    range_profile = np.transpose(range_profile, (1, 0, 2))
    M = range_profile.shape[2]  # Number of range bins

    # Initialize outputs
    condNumber = np.zeros(M)
    rangeAngle = np.zeros((M, steerMat.shape[1]))  # DOA angles
    bfWeight = np.zeros((M, steerMat.shape[0], steerMat.shape[1]))
    invCovMat = np.zeros((M, num_tx * num_rx, num_tx * num_rx))  # Adjust size as needed

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
    return rangeAngle, bfWeight, invCovMat
        
        
def CFAR1D_CASO(signal, K0):
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
    



