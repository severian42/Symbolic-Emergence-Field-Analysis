"""Utility functions for the SEFA library."""

import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Dict, Optional, Union
import logging
import warnings

from .config import SEFAConfig, SavgolParams, PolyfitParams

logger = logging.getLogger(__name__)

def mirror_pad(data: np.ndarray, pad_width: int) -> Tuple[np.ndarray, Dict[str, int]]:
    """Applies mirror padding to 1D data.

    Args:
        data (np.ndarray): The 1D input array.
        pad_width (int): The number of elements to pad on each side.

    Returns:
        Tuple[np.ndarray, Dict[str, int]]:
            - np.ndarray: The padded array.
            - Dict[str, int]: Dictionary containing 'left' and 'right' pad widths used.
    """
    if pad_width <= 0:
        return data, {'left': 0, 'right': 0}
    # Use numpy.pad with mode='reflect' which mirrors edge values
    # Note: 'mirror' in some contexts repeats the edge value, 'reflect' reflects
    # based on center. SEFA.md implies reflection of the signal.
    padded_data = np.pad(data, pad_width, mode='reflect')
    return padded_data, {'left': pad_width, 'right': pad_width}

def handle_boundaries(data: np.ndarray, config: SEFAConfig, original_indices: Optional[np.ndarray]) -> np.ndarray:
    """Applies boundary handling (discarding or slicing padded results).

    This function is typically called *after* an operation like Hilbert transform
    or differentiation that might have boundary artifacts or used padding.

    Args:
        data (np.ndarray): The data potentially containing boundary artifacts or padding.
        config (SEFAConfig): Configuration object with boundary handling settings.
        original_indices (Optional[np.ndarray]): Indices mapping `data` points back to the
                                                 original unpadded/full domain size.
                                                 Used when config.boundary_method was 'mirror'.

    Returns:
        np.ndarray: Data with boundaries handled according to the configuration.
    """
    if config.boundary_method == 'discard':
        n = len(data)
        discard_count = int(n * config.boundary_discard_fraction)
        if discard_count > 0:
            logger.debug(f"Discarding {discard_count} points from each boundary.")
            return data[discard_count : n - discard_count]
        else:
            return data
    elif config.boundary_method == 'mirror':
        if original_indices is not None:
            # If mirror padding was used, `data` is the result on the padded signal.
            # `original_indices` tells us which part corresponds to the original signal.
            logger.debug("Extracting original signal region from mirror-padded result.")
            return data[original_indices]
        else:
            # This case shouldn't happen if mirror padding was applied correctly
            warnings.warn("Boundary method is 'mirror' but no original_indices provided. Returning full data.", UserWarning)
            return data
    elif config.boundary_method == 'periodic':
        # No explicit boundary handling needed for periodic assumption
        return data
    else:
        raise ValueError(f"Unknown boundary_method: {config.boundary_method}")

def _finite_difference(data: np.ndarray, dy: float, order: int) -> np.ndarray:
    """Calculates derivative using central finite differences.

    Handles boundaries using forward/backward differences.
    Note: Prone to noise as per SEFA.md Limitation 2.3.
    """
    if order == 1:
        # Central difference for interior points
        deriv = np.gradient(data, dy, edge_order=1) # Uses 1st order accurate differences at edges
        # SEFA.md: (y[i+1] - y[i-1]) / (2*dy)
        # np.gradient uses this for interior, and adjusts at boundaries.
    elif order == 2:
        # Central difference: (y[i+1] - 2*y[i] + y[i-1]) / dy^2
        # np.gradient with order 2 can approximate this.
        # Alternatively, apply order=1 twice (less accurate).
        # Using np.gradient with edge_order=2 for better boundary accuracy.
        deriv = np.gradient(np.gradient(data, dy, edge_order=2), dy, edge_order=2)
        # Manual implementation would require careful boundary handling.
    else:
        raise ValueError("Finite difference order must be 1 or 2.")
    return deriv

def _savgol_derivative(data: np.ndarray, dy: float, params: SavgolParams) -> np.ndarray:
    """Calculates derivative using Savitzky-Golay filter.

    Recommended method in SEFA.md Limitation 2.3.
    """
    # Ensure window_length is odd and <= data length
    n = len(data)
    window_length = min(params.window_length, n)
    if window_length % 2 == 0:
        window_length -= 1 # Make it odd
    if window_length < params.polyorder + 1 or window_length < 1:
        warnings.warn(f"Adjusted SavGol window_length ({window_length}) is too small "
                       f"for polyorder ({params.polyorder}). Falling back to finite difference.", UserWarning)
        # Fallback or raise error? Falling back for now.
        # Note: Need to handle which deriv order to use in fallback
        return _finite_difference(data, dy, order=params.deriv_order)

    deriv = savgol_filter(
        data,
        window_length=window_length,
        polyorder=params.polyorder,
        deriv=params.deriv_order,
        delta=dy,
        mode='interp' # Handles boundaries by fitting polynomial to available points
                      # Other modes like 'mirror' might align with other boundary handling
    )
    return deriv

def _polyfit_derivative(data: np.ndarray, dy: float, params: PolyfitParams) -> np.ndarray:
    """Calculates derivative using local polynomial fitting (Not fully implemented).

    Recommended method in SEFA.md Limitation 2.3.
    """
    # TODO: Implement local polynomial fitting derivative calculation.
    # This involves:
    # 1. Sliding a window across the data.
    # 2. Fitting a polynomial of degree `params.polyorder` to the data in the window.
    # 3. Analytically calculating the derivative of the fitted polynomial at the window center.
    # 4. Handling boundaries carefully (e.g., using smaller windows or one-sided fits).
    warnings.warn("_polyfit_derivative is not yet implemented. Falling back to Savitzky-Golay.", UserWarning)
    # Fallback to SavGol using corresponding polyfit params
    savgol_params = SavgolParams(
        window_length=params.window_length,
        polyorder=params.polyorder,
        deriv_order=params.deriv_order
    )
    return _savgol_derivative(data, dy, savgol_params)

def calculate_derivative(data: np.ndarray, dy: float, config: SEFAConfig, deriv_order: int) -> np.ndarray:
    """Calculates the specified order derivative using the configured method.

    Args:
        data (np.ndarray): Input data array.
        dy (float): Step size of the domain.
        config (SEFAConfig): Configuration object specifying the derivative method and parameters.
        deriv_order (int): Order of the derivative (1 or 2).

    Returns:
        np.ndarray: The calculated derivative.
    """
    method = config.derivative_method

    if method == 'finite_difference':
        logger.debug(f"Calculating derivative (order={deriv_order}) using finite difference.")
        return _finite_difference(data, dy, order=deriv_order)
    elif method == 'savgol':
        params = config.savgol_frequency_params if deriv_order == 1 else config.savgol_curvature_params
        logger.debug(f"Calculating derivative (order={deriv_order}) using Savitzky-Golay.")
        return _savgol_derivative(data, dy, params)
    elif method == 'polyfit':
        params = config.polyfit_frequency_params if deriv_order == 1 else config.polyfit_curvature_params
        logger.debug(f"Calculating derivative (order={deriv_order}) using polynomial fitting.")
        return _polyfit_derivative(data, dy, params)
    else:
        raise ValueError(f"Unknown derivative_method: {method}")

# TODO (Polyfit Derivative): Implement local polynomial fitting derivative calculation (SEFA.md Limitation 2.3).
# TODO (Phase Unwrapping): Consider adding more robust phase unwrapping algorithms if np.unwrap proves insufficient (SEFA.md Sec II.3.4).
# TODO (Windowing): Implement windowing functions (e.g., Tukey) for optional use with Hilbert transform (SEFA.md Limitation 2.1).
# TODO (Entropy): Implement k-NN / KL estimators for small entropy windows (W < 50) (SEFA.md Limitation 2.2). 