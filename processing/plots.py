import matplotlib.pyplot as plt
import os


def _save_figure(fig, filename):
    # save the current figure into the output directory
    path = os.path.join('output', filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_approximation(x, y_true, y_pred, filename='approximation_all_data.png'):
    # original function vs RBF approximation using all data
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y_true, 'b-', label='Original function')
    ax.plot(x, y_pred, 'r--', label='RBF approximation')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Function Approximation with RBF Network (all data)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_figure(fig, filename)


def plot_reconstruction(x, y_true, y_observed, y_pred, removal_pct, filename):
    # original, data with missing values, and RBF reconstruction
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y_true, 'g-', linewidth=2, label='Original function')
    ax.plot(x, y_observed, 'bo', markersize=3, label='Data with missing values')
    ax.plot(x, y_pred, 'r--', linewidth=2, label='RBF reconstruction')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Function Reconstruction with RBF ({}% removed)'.format(removal_pct))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_figure(fig, filename)


def plot_comparison(results, filename='reconstruction_comparison.png'):
    # compare reconstruction quality across removal percentages
    percents = [r['removal_pct'] for r in results]
    mses = [r['mse'] for r in results]
    rmses = [r['rmse'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(percents, mses, 'o-', color='tab:red')
    axes[0].set_xlabel('Data removed (%)')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE vs Data Removed')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(percents, rmses, 'o-', color='tab:blue')
    axes[1].set_xlabel('Data removed (%)')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE vs Data Removed')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, filename)