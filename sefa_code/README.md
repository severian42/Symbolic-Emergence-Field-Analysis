# SEFA - Symbolic Emergence Field Analysis

Python implementation of the Symbolic Emergence Field Analysis (SEFA) algorithm, derived from the first-principles description in SEFA.md.

## Overview

SEFA provides a self-calibrating framework to detect emergent symbolic structures within a constructed field. It analyzes local geometric (amplitude, curvature, frequency) and information-theoretic (entropy) features to produce a score highlighting regions of potential significance.

## Installation

```bash
# Install from PyPI (once published)
# pip install sefa

# Install from source
pip install .

# Install with development dependencies (for examples and tests)
pip install ".[dev]"
```

## Quickstart

```python
import numpy as np
from sefa import SEFA
import matplotlib.pyplot as plt

# 1. Define domain and drivers
y_min = 0
y_max = 2 * np.pi
M = 500
drivers = np.array([2, 5, 13]) # Example drivers

# 2. Initialize and run SEFA
sefa_analyzer = SEFA(y_min=y_min, y_max=y_max, M=M, drivers=drivers)
sefa_score = sefa_analyzer.compute()

# 3. Analyze results
domain = sefa_analyzer.domain
v0 = sefa_analyzer.get_feature('V0')
amplitude = sefa_analyzer.get_feature('A')
symbol_indices, threshold = sefa_analyzer.detect_symbols(method='otsu')
symbol_locs = domain[symbol_indices]

# 4. Plot (optional)
plt.figure(figsize=(10, 6))
plt.plot(domain, v0, label='V0(y)', alpha=0.5)
plt.plot(domain, amplitude, label='A(y)', linestyle='--')
plt.plot(domain, sefa_score, label='SEFA(y)', color='red', linewidth=2)
plt.scatter(symbol_locs, sefa_score[symbol_indices], color='black', zorder=5, label='Detected Symbols')
plt.axhline(threshold, color='gray', linestyle=':', label=f'Otsu Threshold ({threshold:.2f})')
plt.title('SEFA Example')
plt.xlabel('Domain (y)')
plt.ylabel('Field Value / Score')
plt.legend()
plt.grid(True)
plt.show()

print(f"Detected {len(symbol_locs)} symbols at y = {symbol_locs}")
```

## Documentation

Refer to the docstrings within the code and the original `SEFA.md` document for detailed explanations of the algorithm, features, and parameters.

## Testing

Run tests using pytest:

```bash
pytest tests/
```

## Contributing

Contributions are welcome. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) (to be created).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

python examples/run_sefa_example.py

python sefa_ml.py