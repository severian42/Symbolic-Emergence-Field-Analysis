"""Functions for field construction and analytic signal generation (Hilbert Transform)."""

import numpy as np
from scipy.signal import hilbert
from typing import Tuple, Optional
import logging
import warnings

from .config import SEFAConfig
from .utils import mirror_pad # Import padding utility

logger = logging.getLogger(__name__)

def construct_field(domain_y: np.ndarray, gamma_k: np.ndarray, weights_w: np.ndarray) -> np.ndarray:
    """Constructs the real-valued field V0(y) = Sum[w_k * Cos(gamma_k * y)].

    Implements SEFA.md Section II.2.

    Args:
        domain_y (np.ndarray): Discretized domain points.
        gamma_k (np.ndarray): Driver parameters.
        weights_w (np.ndarray): Corresponding driver weights.

    Returns:
        np.ndarray: The constructed field V0.
    """
    if len(gamma_k) != len(weights_w):
        raise ValueError("Number of drivers (gamma_k) must match number of weights (weights_w).")

    # Use broadcasting for efficient computation
    # V0 = Sum_k [ w_k * cos(gamma_k * y_i) ]
    # Reshape for broadcasting: y (M,), gamma (K,), w (K,)
    # Outer product gamma_k * y_i gives shape (K, M)
    # Cosine is element-wise: shape (K, M)
    # Multiply by weights (broadcast w_k along M): shape (K, M)
    # Sum along k axis (axis=0): shape (M,)
    V0 = np.sum(weights_w[:, np.newaxis] * np.cos(gamma_k[:, np.newaxis] * domain_y), axis=0)
    return V0

def get_analytic_signal(field_V0: np.ndarray, config: SEFAConfig) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Calculates the analytic signal V0 + i*H(V0) using the Hilbert transform.

    Implements SEFA.md Section II.3.1 and II.3.2.
    Handles boundary artifacts based on config.boundary_method (Limitation 2.1).

    Args:
        field_V0 (np.ndarray): The real-valued input field.
        config (SEFAConfig): Configuration object with boundary handling settings.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]:
            - np.ndarray: The complex analytic signal.
            - Optional[np.ndarray]: Indices corresponding to the original signal
                                    if padding was used (needed for boundary handling later).
                                    Returns None if no padding/cropping occurred relative to input.
    """
    original_indices = np.arange(len(field_V0))
    signal_to_transform = field_V0
    padding_info = None

    if config.boundary_method == 'mirror':
        # Apply mirror padding before Hilbert transform
        # TODO: Windowing (e.g., Tukey) could also be applied here or before padding
        #       as mentioned in Limitation 2.1, but sticking to padding for now.
        pad_width = int(len(field_V0) * config.boundary_discard_fraction) # Use fraction as guide for padding size
        if pad_width > 0:
            logger.debug(f"Applying mirror padding width: {pad_width}")
            padded_signal, padding_info = mirror_pad(field_V0, pad_width)
            signal_to_transform = padded_signal
        else:
            warnings.warn("Mirror padding requested, but padding fraction results in zero padding width. Skipping.", UserWarning)

    elif config.boundary_method == 'discard':
        # No padding needed here, discarding happens *after* transform in handle_boundaries (utils.py)
        pass
    elif config.boundary_method == 'periodic':
        # Standard FFT-based Hilbert assumes periodicity, no explicit action needed before transform.
        logger.warning("Using periodic boundary assumption for Hilbert transform. "
                       "Beware of potential edge artifacts (SEFA.md Limitation 2.1).")
    else:
         raise ValueError(f"Unknown boundary_method: {config.boundary_method}")

    # Calculate analytic signal using scipy.fft.hilbert
    # This directly computes V0 + i*H(V0)
    analytic_signal = hilbert(signal_to_transform)

    # If padded, return the full padded signal and the original indices slice
    if padding_info is not None:
        # We need indices relative to the *padded* signal that correspond to the original signal
        original_indices_in_padded = np.arange(padding_info['left'], padding_info['left'] + len(field_V0))
        return analytic_signal, original_indices_in_padded
    else:
        # If no padding, the returned signal corresponds directly to original indices
        return analytic_signal, original_indices # Or None, as indices match 1:1?
                                              # Returning original_indices is safer for handle_boundaries

# TODO (Windowing): Implement optional windowing (e.g., Tukey) before Hilbert transform 
#                   as mentioned in SEFA.md Limitation 2.1.

# TODO: Implement functions as per SEFA.md Section II.3.1, II.3.2
# TODO: Address Limitation 2.1 (Boundary Treatment) - padding/windowing 