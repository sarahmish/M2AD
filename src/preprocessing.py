import numpy as np

def sliding_window_sequences(X, index, y=None, window_size=100, step_size=1, target_size=1):
    """Generate sliding windows from given input.

    Args:
        X (ndarray):
            2-d array of input sequences.
        index (array):
            List of index values of the input sequence, typically timestamps.
        y (ndarray):
            2-d array of target sequences. If not given, use `X`.
        window_size (int):
            Size of the window. Default `100`.
        step_size (int):
            Stride size. Default `1`.
        target_size (int):
            Size of the prediction horizon. Default `1`.
    
    Returns:
        ndarray, ndarray:
            * 3-d array of input sequences.
            * 2-d array of target sequences.
            * 1-d array of index values of the first target.
    """
    # TODO: make this function generalizable for long horizon.
    if y is None:
        y = X.copy()

    windows = list()
    targets = list()
    indices = list()

    length = len(X)
    for i in range(0, length - window_size - 1, step_size):
        start = i
        end = i + window_size
        windows.append(X[start: end])
        targets.append(y[end + 1])
        indices.append(index[end + 1])

    return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(indices)
