from data.generator import generate_data, remove_random_data
from processing.model import rbf_train, rbf_predict
from processing.metrics import compute_metrics


def run_reconstruction(removal_percentages=(20, 60, 85), n_centers=30, spread=1.0, seed=42):
    # generate the full function data
    x, y = generate_data()
    results = []
    for pct in removal_percentages:
        # mark a percentage of values as missing
        y_missing, _, observed_mask = remove_random_data(x, y, pct, seed)
        # train RBF only on the observed (non-missing) data
        x_train = x[observed_mask]
        y_train = y[observed_mask]
        centers, weights, spread = rbf_train(x_train, y_train, n_centers, spread, seed)
        # predict on the full grid including missing points
        y_pred = rbf_predict(x, centers, weights, spread)
        # compute metrics on observed points only (fair comparison)
        metrics = compute_metrics(y[observed_mask], y_pred[observed_mask])
        metrics['removal_pct'] = pct
        metrics['x'] = x
        metrics['y'] = y
        metrics['y_missing'] = y_missing
        metrics['y_pred'] = y_pred
        metrics['observed_mask'] = observed_mask
        results.append(metrics)
    return results