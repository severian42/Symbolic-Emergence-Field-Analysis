# SEFA - Symbolic Emergence Field Analysis (Python Implementation)

This directory contains the core Python implementation of the Symbolic Emergence Field Analysis (SEFA) algorithm. SEFA is designed to analyze the information geometry and complexity of continuous fields constructed from discrete driver data, identifying regions of potential emergent symbolic significance.

## Overview

The SEFA algorithm provides a framework for:
1.  Constructing a potential field (`V0`) based on input "driver" data points (`gamma_k`) over a specified domain.
2.  Calculating local geometric features (Amplitude, Frequency, Curvature) and information-theoretic features (local Entropy, Entropy Alignment) of this field.
3.  Combining these features using derived weights based on their global information content (Information Deficits).
4.  Producing a final SEFA score that highlights regions of high combined complexity or "emergence".
5.  Applying thresholding techniques to detect potential symbolic locations based on the SEFA score.

This implementation is modular, breaking down the SEFA pipeline into distinct components.

## Core Components (`sefa/` directory)

The implementation is organized into the following key modules:

*   `__init__.py`: Makes the primary `SEFA` class and configuration objects easily importable.
*   `config.py`: Defines configuration classes (`SEFAConfig`, `SavgolParams`, etc.) allowing customization of algorithm parameters (e.g., derivative methods, window sizes, weighting factors).
*   `drivers.py`: Handles processing and weighting of the input driver data (`gamma_k`).
*   `domain.py`: Manages the discretization of the analysis domain (`y`).
*   `transform.py`: Computes the base potential field (`V0`) from the weighted drivers and performs the Hilbert transform for analytic signal calculation.
*   `features.py`: Calculates local geometric features (Amplitude `A`, Frequency `F`, Curvature `C`) from the analytic signal.
*   `entropy.py`: Calculates local information-theoretic features (Entropy `S`, Entropy Alignment `E`).
*   `scoring.py`: Computes information deficits (`w_X`), feature exponents (`alpha_X`), and combines the normalized features into the final SEFA score.
*   `thresholding.py`: Implements methods (e.g., Otsu, percentile) to automatically detect significant points ("symbols") in the SEFA score.
*   `sefa_core.py`: Orchestrates the entire SEFA pipeline, integrating the steps from configuration and input processing through feature calculation, scoring, and providing access to results.
*   `utils.py`: Contains helper functions used across various modules (e.g., normalization, boundary handling, derivative calculations).

## Usage Example

```python
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from sefa_code.sefa import SEFA, SEFAConfig # Import main class and config

# 1. Define Drivers and Domain Parameters
drivers_gamma = np.array([-13.6, -3.4, -1.51]) # Example: First 3 Hydrogen levels (eV)
ymin = -15.0
ymax = 0.0
num_points = 1000

# 2. Configure SEFA (Optional - defaults are provided)
# See sefa/config.py for all options
config = SEFAConfig(
    beta=2.0,
    p_features=4,
    derivative_method='savgol',
    entropy_window_size=51, # Should be odd
    boundary_method='discard',
    boundary_discard_fraction=0.05
)

# 3. Initialize and Run SEFA Pipeline
# Configuration can be passed at initialization
sefa_analyzer = SEFA(config=config)

# Run the full analysis pipeline
sefa_analyzer.run_pipeline(
    drivers_gamma=drivers_gamma,
    ymin=ymin,
    ymax=ymax,
    num_points=num_points
)

# 4. Get Results
results = sefa_analyzer.get_results()
sefa_score = results['sefa_score']
processed_domain_y = results['processed_domain_y']
# Access individual features if needed:
# amplitude_A = results.get('amplitude_A')
# entropy_S = results.get('entropy_S')
# exponents = results.get('exponents_alpha')

# 5. Detect Symbols (Optional)
# Uses settings from config unless overridden
mask = sefa_analyzer.threshold_score(method='percentile', percentile=95)
symbol_indices = np.where(mask)[0]
symbol_locs = processed_domain_y[symbol_indices]
threshold_value = results.get('threshold_value') # Actual threshold used

print(f"SEFA analysis complete. Found {len(symbol_locs)} potential symbols.")
print(f"Symbol locations (y): {np.round(symbol_locs, 4)}")

# --- Plotting (Example using matplotlib) ---
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(processed_domain_y, sefa_score, label='SEFA Score', color='red', zorder=10)
ax.scatter(symbol_locs, sefa_score[symbol_indices], color='black', marker='o', s=50, zorder=11, label=f'Symbols ({len(symbol_locs)})')
if threshold_value is not None:
    ax.axhline(threshold_value, color='gray', linestyle=':', label=f'95th Percentile Thresh ({threshold_value:.3f})')

# Overlay drivers
ax_twin = ax.twinx()
ax_twin.stem(drivers_gamma, results['weights_w'], linefmt='C0--', markerfmt='C0o', basefmt=' ', label='Driver Weights')
ax_twin.set_ylabel('Driver Weight', color='C0')
ax_twin.tick_params(axis='y', labelcolor='C0')
ax_twin.spines['right'].set_color('C0')

ax.set_xlabel('Domain (y)')
ax.set_ylabel('SEFA Score', color='red')
ax.tick_params(axis='y', labelcolor='red')
ax.set_title('SEFA Analysis Example')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.show()

```

## Installation

Install the package from the `sefa_code` directory:

```bash
# Install core package
pip install .

# Install with development dependencies (for examples, tests)
pip install ".[dev]"
```

## Dependencies

This package requires standard scientific Python libraries:
*   `numpy`
*   `scipy`

Development dependencies (for examples/tests) may include:
*   `matplotlib`
*   `pytest`
*   `numba` (Optional, used in some examples for performance)
*   `networkx` (Optional, used in some examples)
*   `plotly` (Optional, used in some examples)
*   `pyvis` (Optional, used in some examples)

Refer to `setup.py` or `requirements.txt` for precise dependencies.

## Relationship to `SEFA.md`

This code implements the algorithmic principles and mathematical framework detailed in the `SEFA.md` document (located in the parent directory or project root). Refer to `SEFA.md` for the theoretical background and first-principles derivation.

## Testing

Ensure tests pass before committing changes. Run tests from the `sefa_code` directory using pytest:

```bash
pytest tests/
```

## Contributing

Contributions are welcome. Please outline your proposed changes in an issue or pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
