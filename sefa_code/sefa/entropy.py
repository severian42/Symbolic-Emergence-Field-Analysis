"""Module for calculating sliding window entropy and alignment."""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from astropy.stats import knuth_bin_width # For adaptive binning
from numpy.lib.stride_tricks import sliding_window_view
import logging

from .config import SEFAConfig

logger = logging.getLogger(__name__)

def _calculate_local_entropy(window_data: np.ndarray, config: SEFAConfig) -> float:
    """Calculates entropy for a single window using the configured method."""
    # Remove NaNs if any exist in the window
    window_data = window_data[~np.isnan(window_data)]
    if window_data.size < 2:
        # Cannot calculate entropy reliably with < 2 points
        return np.nan # Or 0? NaN indicates inability to compute

    bin_method = config.entropy_binning
    bins: Union[int, str, np.ndarray] = 'auto' # Default for np.histogram

    if bin_method == 'knuth':
        # Calculate bin width using Knuth's rule
        # Requires astropy
        try:
            bin_width, bins = knuth_bin_width(window_data, return_bins=True)
            # knuth_bin_width can return 0 bins if data is constant
            if len(bins) <= 1:
                 logger.debug(f"Knuth rule resulted in <= 1 bin for window size {len(window_data)}. Using fixed bins=2 as fallback.")
                 bins = 2 # Fallback to minimum meaningful bins
        except Exception as e:
            logger.warning(f"Knuth bin calculation failed: {e}. Falling back to fixed bins ({config.entropy_bin_count}).")
            bins = config.entropy_bin_count

    elif bin_method == 'fixed':
        bins = config.entropy_bin_count
    else:
        raise ValueError(f"Unknown entropy_binning method: {bin_method}")

    # Calculate histogram
    # Use density=True to get probability density, or False for counts?
    # SEFA.md uses q_j, implying probabilities. scipy.stats.entropy expects counts.
    # Let's use counts and scipy.stats.entropy for direct pk*log(pk) calculation.
    try:
        counts, bin_edges = np.histogram(window_data, bins=bins)
        # Filter out zero counts as they don't contribute to entropy and cause log(0)
        counts = counts[counts > 0]
        if counts.size == 0:
             # Handle cases where all data falls into one bin or window is empty after NaN removal
             return 0.0 # Zero entropy for constant signal within window

        # Calculate entropy using scipy.stats.entropy (calculates -sum(pk * log(pk)))
        # It automatically handles normalization to probabilities pk.
        local_entropy = scipy_entropy(counts, base=np.e) # Use natural logarithm (base e)

    except ValueError as ve:
         # Catch potential errors from np.histogram (e.g., if bins are invalid)
         logger.error(f"Error during histogram calculation for entropy: {ve}")
         local_entropy = np.nan

    return local_entropy

def calculate_sliding_window_entropy(data: np.ndarray, config: SEFAConfig) -> np.ndarray:
    """Calculates local entropy using a sliding window.

    Implements SEFA.md Section II.4.a.
    Handles small window sizes (Limitation 2.2) by falling back to histogram
    until specific estimators (k-NN, KL) are implemented.

    Args:
        data (np.ndarray): Input data (typically Amplitude A(y)).
        config (SEFAConfig): Configuration object with entropy settings.

    Returns:
        np.ndarray: Array of local entropy values S(y).
    """
    window_size = config.entropy_window_size
    if window_size > len(data):
        raise ValueError(f"entropy_window_size ({window_size}) cannot be larger than data length ({len(data)}).")
    if window_size < 2:
        raise ValueError(f"entropy_window_size ({window_size}) must be at least 2.")

    # TODO (Limitation 2.2): Implement specific estimators for small W < 50
    # if window_size < 50 and config.small_window_entropy_method == 'knn':
    #     logger.info("Using k-NN entropy estimator for small window size (Not Implemented).")
    #     # Call k-NN entropy function here
    # elif window_size < 50 and config.small_window_entropy_method == 'kl':
    #     logger.info("Using KL entropy estimator for small window size (Not Implemented).")
    #     # Call KL entropy function here

    logger.debug(f"Calculating sliding window entropy with window size {window_size} and binning '{config.entropy_binning}'")

    # Use sliding_window_view for efficiency
    windows = sliding_window_view(data, window_shape=window_size)

    # Apply entropy calculation to each window
    # Vectorization is tricky here due to adaptive binning per window.
    # Loop approach:
    entropy_values = np.full(len(data), np.nan) # Initialize with NaNs
    center_offset = window_size // 2

    for i, window in enumerate(windows):
        idx = i + center_offset # Index corresponding to the center of the window
        entropy_values[idx] = _calculate_local_entropy(window, config)

    # Handle boundaries where the window didn't fit completely (NaNs remain)
    # Option 1: Keep NaNs (simplest, user handles downstream)
    # Option 2: Extrapolate/interpolate (complex)
    # Option 3: Calculate with smaller/asymmetric windows at edges (complex)
    # Keeping NaNs for now.
    num_nans = np.sum(np.isnan(entropy_values))
    if num_nans > 0:
        logger.debug(f"{num_nans} NaN values at boundaries due to sliding window entropy calculation.")

    return entropy_values

def calculate_entropy_alignment(entropy_S: np.ndarray) -> np.ndarray:
    """Calculates the entropy alignment score E(y) = 1 - S(y)/SMax.

    Implements SEFA.md Section II.4.b.

    Args:
        entropy_S (np.ndarray): Array of local entropy values.

    Returns:
        np.ndarray: Array of entropy alignment scores.
    """
    # Find SMax, ignoring potential NaNs at boundaries
    if np.all(np.isnan(entropy_S)):
        logger.warning("All entropy values are NaN. Cannot calculate SMax. Returning NaNs.")
        return entropy_S # Return original NaNs

    SMax = np.nanmax(entropy_S)

    if SMax is None or SMax == 0 or not np.isfinite(SMax):
        logger.warning(f"Could not determine valid SMax (found: {SMax}). Entropy alignment calculation might be invalid. Returning zeros.")
        # Avoid division by zero or NaN/Inf propagation
        # If SMax is 0, all S are likely 0, so alignment is 1. Handle NaN case.
        alignment_E = np.where(np.isnan(entropy_S), np.nan, 1.0 if SMax == 0 else 0.0)
        return alignment_E

    # Calculate alignment, preserving NaNs where entropy was NaN
    alignment_E = 1.0 - (entropy_S / SMax)
    # Ensure alignment is non-negative (should be, unless S < 0 or S > SMax due to numerical issues)
    alignment_E = np.maximum(0, alignment_E)

    return alignment_E 