from data.generator import generate_data
from tasks.baseline import run_baseline
from tasks.reconstruction import run_reconstruction
from processing.plots import plot_approximation, plot_reconstruction, plot_comparison


def main():
    # configuration parameters
    n_centers = 30
    spread = 1.0
    seed = 42
    removal_percentages = (20, 60, 85)

    print('Radial Basis Function Networks - Function Approximation')
    print()

    # generate data info
    x, y = generate_data()
    print('Generating data...')
    print('  Range: -2*pi to 2*pi | Step: 0.01 | Samples: {}'.format(len(x)))
    print()

    # Task 1: approximate with all data
    print('Running Task 1: Function Approximation (all data)...')
    print('  Centers: {} | Spread: {}'.format(n_centers, spread))
    baseline = run_baseline(n_centers=n_centers, spread=spread, seed=seed)
    print('    MSE: {:.4f}'.format(baseline['mse']))
    print('    RMSE: {:.4f}'.format(baseline['rmse']))
    print('    MAE: {:.4f}'.format(baseline['mae']))
    print()
    plot_approximation(baseline['x'], baseline['y'], baseline['y_pred'])

    # Task 2: reconstruct with missing data
    print('Running Task 2: Function Reconstruction (missing data)...')
    results = run_reconstruction(removal_percentages=removal_percentages,
                                 n_centers=n_centers, spread=spread, seed=seed)
    for res in results:
        print('  {}% removed:'.format(res['removal_pct']))
        print('    MSE: {:.4f}'.format(res['mse']))
        print('    RMSE: {:.4f}'.format(res['rmse']))
        plot_reconstruction(res['x'], res['y'], res['y_missing'], res['y_pred'],
                            res['removal_pct'],
                            filename='reconstruction_{}.png'.format(res['removal_pct']))
    print()
    plot_comparison(results)

    print('Done. Figures saved in output/.')


if __name__ == '__main__':
    main()