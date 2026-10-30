import numpy as np


def target_function(x):
    # compute f(x) = 2cos(2x) - sin(x) + 0.9 * exp(sin(3x))
    y = 2 * np.cos(2 * x) - np.sin(x) + 0.9 * np.exp(np.sin(3 * x))
    return y


def generate_data(start=-2 * np.pi, end=2 * np.pi, step=0.01):
    # create the x grid from -2*pi to 2*pi with the given step
    x = np.arange(start, end + step, step)
    # evaluate the target function on the full grid
    y = target_function(x)
    return x, y


def remove_random_data(x, y, removal_percentage, seed=42):
    # randomly mark a percentage of values as missing (NaN)
    rng = np.random.RandomState(seed)
    n_total = len(y)
    n_missing = int(round(removal_percentage / 100.0 * n_total))
    missing_indices = rng.choice(n_total, size=n_missing, replace=False)
    y_missing = y.copy()
    y_missing[missing_indices] = np.nan
    # boolean mask of observed (non-missing) values
    observed_mask = ~np.isnan(y_missing)
    return y_missing, missing_indices, observed_mask