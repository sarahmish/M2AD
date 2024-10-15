import numpy as np

def sliding_window_sequences(X, index, y=None, window_size=100, step_size=1, target_size=1):
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
