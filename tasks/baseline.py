from data.generator import generate_data
from processing.model import rbf_train, rbf_predict
from processing.metrics import compute_metrics


def run_baseline(n_centers=30, spread=1.0, seed=42):
    # generate the full function data
    x, y = generate_data()
    # train RBF on all available data
    centers, weights, spread = rbf_train(x, y, n_centers, spread, seed)
    # predict on the full grid
    y_pred = rbf_predict(x, centers, weights, spread)
    # compute error metrics on the full dataset
    metrics = compute_metrics(y, y_pred)
    metrics['x'] = x
    metrics['y'] = y
    metrics['y_pred'] = y_pred
    metrics['centers'] = centers
    metrics['spread'] = spread
    metrics['n_centers'] = n_centers
    return metrics