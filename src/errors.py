from functools import partial

import pandas as pd
import numpy as np

def _smooth(errors, smoothing_window):
    smoothed_errors = pd.DataFrame(errors).ewm(smoothing_window).mean().values
    
    # mean value for all entries <= window_size
    # errors[:smoothing_window] = np.mean(errors[:smoothing_window*2]).repeat(smoothing_window)

    return smoothed_errors


def point_errors(y, pred, smooth=False, smoothing_window=10):
    """Compute point-wise difference.
    
    Args:
        y (array):
            1-dimensional array of original sequences.
        pred (array):
            1-dimensional array of predicted sequences.
        smooth (bool):
            Whether to apply exponential smoothing. Default False.
        smoothing_window (int):
            Size of smoothing window if applied.

    Returns:
        array:
            1-dimensional array of computed error values.
    """
    errors = np.abs(y - pred)

    if smooth:
        errors = _smooth(errors, smoothing_window)

    return np.array(errors)


def area_errors(y, pred, score_window=10, dx=100, smooth=False, smoothing_window=10):
    """Compute area difference.
    
    Args:
        y (array):
            1-dimensional array of original sequences.
        pred (array):
            1-dimensional array of predicted sequences.
        score_window (int):
            Length of window to compute integral over.
        dx (int):
            The spacing between sample points.
        smooth (bool):
            Whether to apply exponential smoothing. Default False.
        smoothing_window (int):
            Size of smoothing window if applied.

    Returns:
        array:
            1-dimensional array of computed error values.
    """
    trapz = partial(np.trapz, dx=dx)
    errors = np.empty_like(y)
    num_signals = errors.shape[1]

    for i in range(num_signals):
        area_y = pd.Series(y[:, i]).rolling(
            score_window, center=True, min_periods=score_window // 2).apply(trapz)
        area_pred = pd.Series(pred[:, i]).rolling(
            score_window, center=True, min_periods=score_window // 2).apply(trapz)
        
        error = area_y - area_pred

        if smooth:
            error = _smooth(error, smoothing_window)
        
        errors[:, i] = error.flatten()

    mu = np.mean(errors)
    std = np.std(errors)

    return (errors - mu) / std