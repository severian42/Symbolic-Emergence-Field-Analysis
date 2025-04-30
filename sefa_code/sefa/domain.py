"""Functions for domain discretization and driver handling."""

import numpy as np
from typing import Tuple


def discretize_domain(ymin: float, ymax: float, num_points: int) -> Tuple[np.ndarray, float]:
    """Discretizes the domain [ymin, ymax] into M points.

    Implements SEFA.md Section II.1.

    Args:
        ymin (float): Minimum value of the domain.
        ymax (float): Maximum value of the domain.
        num_points (int): Number of points (M) to create.

    Returns:
        Tuple[np.ndarray, float]: A tuple containing:
            - np.ndarray: The array of discretized domain points (y_i).
            - float: The step size (dy).

    Raises:
        ValueError: If num_points is less than 2.
    """
    if num_points < 2:
        raise ValueError("Number of points (num_points) must be at least 2 for discretization.")
    domain_y = np.linspace(ymin, ymax, num_points)
    # dy = (ymax - ymin) / (num_points - 1) # Calculate dy precisely as in SEFA.md
    dy = domain_y[1] - domain_y[0] if num_points > 1 else 0.0
    # Verify the last point is correct
    # assert np.isclose(domain_y[-1], ymax), f"linspace end point mismatch: {domain_y[-1]} vs {ymax}"
    # Verify dy consistency
    # if num_points > 1:
        # assert np.isclose(dy, (ymax - ymin) / (num_points - 1)), f"dy mismatch: {dy} vs {(ymax - ymin) / (num_points - 1)}"

    return domain_y, dy 