import numpy as np


def gaussian_basis(x, centers, spread):
    # compute Gaussian RBF activations between inputs and centers
    # x: (n,), centers: (m,)
    diff = x[:, None] - centers[None, :]
    dist_sq = diff ** 2
    activations = np.exp(-dist_sq / (spread ** 2))
    return activations


def rbf_train(x_train, y_train, n_centers, spread, seed=42):
    # select centers randomly from the training inputs
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(x_train), size=n_centers, replace=False)
    centers = x_train[indices]
    # compute hidden layer activations for training data
    phi = gaussian_basis(x_train, centers, spread)
    # add bias column to the design matrix
    phi_bias = np.column_stack([np.ones(len(x_train)), phi])
    # solve for output weights using least squares (pseudoinverse)
    weights, _, _, _ = np.linalg.lstsq(phi_bias, y_train, rcond=None)
    return centers, weights, spread


def rbf_predict(x, centers, weights, spread):
    # compute hidden activations for the inputs to predict
    phi = gaussian_basis(x, centers, spread)
    # add bias column
    phi_bias = np.column_stack([np.ones(len(x)), phi])
    # linear combination of basis functions with learned weights
    y_pred = phi_bias @ weights
    return y_pred