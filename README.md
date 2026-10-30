# Radial Basis Function Networks - Function Approximation
This project approximates and reconstructs a mathematical function using a
Radial Basis Function (RBF) network implemented from scratch with numpy.

The target function is:

$f(x) = 2cos(2x) - sin(x) + 0.9 * exp(sin(3x))$

defined on the interval $-2*pi <= x <= 2*pi$ with a step of 0.01.


## Tasks 

The work is split into two tasks:

- Task 1: approximate the function using ALL the data. The RBF network must
  reproduce the same function it was taught. Error metrics (MSE, RMSE, MAE)
  are reported on the full dataset.
- Task 2: reconstruct the same function with random data removed at 20%,
  60% and 85%. For each level the original function, the function with missing
  values, and the RBF reconstruction are shown. Error metrics are reported
  for each removal level.

## Function format

The data is generated programmatically by
`data/generator.py`:

- x ranges from -2*pi to 2*pi with step 0.01 (1257 samples).
- y = f(x) evaluated on the full grid.
- Missing values are removed randomly using a fixed seed for reproducibility.

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```


## How to run

Run the only entry point from the project root:

```bash
python main.py
```

All figures are saved to the `output/` folder.

## Parameters / variables to tweak

Edit `main.py` to adjust these variables:

- n_centers: number of RBF centers (default: 30)
- spread: Gaussian spread parameter (default: 1.0)
- seed: random seed for data removal and center selection (default: 42)
- removal_percentages: tuple of removal levels (default: (20, 60, 85))

Edit `data/generator.py` to change:

- the function formula in `target_function`
- the range and step in `generate_data`

## Expected output

The following figures are saved in `output/`:

- approximation_all_data.png : Task 1, original vs RBF approximation
- reconstruction_20.png : Task 2, 20% data removed
- reconstruction_60.png : Task 2, 60% data removed
- reconstruction_85.png : Task 2, 85% data removed
- reconstruction_comparison.png : MSE and RMSE across removal levels

Terminal output shows the error metrics for each task and removal level.