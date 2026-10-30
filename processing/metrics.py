import numpy as np


def mse(y_true, y_pred):
    # mean squared error between true and predicted values
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    # root mean squared error
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    # mean absolute error
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    # coefficient of determination R^2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def compute_metrics(y_true, y_pred):
    # gather all regression metrics into a dictionary
    return {
        'mse': mse(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }