def sliding_window(data, window_size=30, step_size=1):
    """
    Apply a sliding window over the data.

    Parameters:
    - data: The input data with shape (frames, tx, rx, range).
    - window_size: The size of each window/frame slice. Default is 30.
    - step_size: The step size/increment for the sliding window. Default is 1.

    Returns:
    - A list of data slices with shape (window_size, tx, rx, range).
    """
    num_frames = data.shape[0]
    slices = []

    for start in range(0, num_frames - window_size + 1, step_size):
        end = start + window_size
        slices.append(data[start:end])

    return slices