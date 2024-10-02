import numpy as np

# Create a function to save filenames to a text file
def save_filenames_to_txt(filenames, file_path):
    with open(file_path, 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")
            
# Function to get indices of filenames in the original file_path_list
def get_indices(filenames, dataset):
    """Get the indices from the dataset corresponding to the filenames."""
    # Strip both the filenames and the entries in the dataset file path list
    file_path_list_stripped = [fp.strip() for fp in dataset.file_path_list]
    return [file_path_list_stripped.index(filename.strip()) for filename in filenames]

# Function to load filenames from a text file
def load_filenames_from_txt(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]
    
    
    
def fill_based_on_context(arr):
    arr = arr.astype(float)  # Ensure array is float type
    n = len(arr)
    
    # Check the majority value in the array
    majority_value = 1.0 if np.sum(arr) >= n / 2 else 0.0
    
    # Fill left boundary if isolated
    if arr[0] != majority_value and arr[1] == majority_value:
        arr[0] = majority_value
    
    # Fill right boundary if isolated
    if arr[-1] != majority_value and arr[-2] == majority_value:
        arr[-1] = majority_value
    
    return arr

    
    
def smooth_signal(arr, window_size=5):
    """
    Smooths the input signal using a moving average filter and applies a threshold.

    Parameters:
    arr (numpy.ndarray): The input array to be smoothed.
    window_size (int, optional): The size of the moving average window. Default is 3.

    Returns:
    numpy.ndarray: The smoothed array with values thresholded to 0 or 1.
    """
    # Create a moving average kernel
    kernel = np.ones(window_size) / window_size
    # Convolve the input array with the kernel
    smoothed_arr = np.convolve(arr, kernel, mode='same')
    # Apply threshold to determine if value should be 0 or 1
    smoothed_arr = np.where(smoothed_arr >= 0.5, 1, 0)
    smoothed_arr = fill_based_on_context(smoothed_arr)
    return np.where(smoothed_arr >= 0.5, 1, 0)
