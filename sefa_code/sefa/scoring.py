"""Functions for feature normalization, weighting, and final SEFA score calculation."""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from astropy.stats import knuth_bin_width # For adaptive binning of global features
from typing import Dict, List, Union
import logging
import warnings

from .config import SEFAConfig

logger = logging.getLogger(__name__)

def normalize_features(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalizes features according to SEFA.md Section II.5.a.

    X_prime(y) = X(y) / MaxValue(|X(y)|)
    Handles Amplitude (A), Curvature (C), Frequency (F), Entropy Alignment (E).
    Preserves sign for C and F.

    Args:
        features (Dict[str, np.ndarray]): Dictionary of raw feature arrays.

    Returns:
        Dict[str, np.ndarray]: Dictionary of normalized feature arrays.
    """
    normalized = {}
    for name, data in features.items():
        if np.all(np.isnan(data)):
            logger.warning(f"Feature '{name}' contains only NaNs. Normalization results in NaNs.")
            normalized[name] = data # Keep NaNs
            continue

        # Calculate max absolute value, ignoring NaNs
        max_abs_val = np.nanmax(np.abs(data))

        if max_abs_val is None or max_abs_val == 0 or not np.isfinite(max_abs_val):
            logger.warning(f"Could not normalize feature '{name}' (max_abs_val={max_abs_val}). Returning zeros.")
            # Avoid division by zero or NaN/Inf. Return array of zeros preserving NaNs.
            normalized[name] = np.where(np.isnan(data), np.nan, 0.0)
        else:
            # Normalize, preserving original NaNs
            normalized[name] = np.where(np.isnan(data), np.nan, data / max_abs_val)

            # Special check for A and E which should be non-negative post-normalization
            # (They are non-negative before, but floating point issues might occur)
            if name in ['A', 'E']:
                normalized[name] = np.maximum(0, normalized[name])

            logger.debug(f"Normalized feature '{name}' using max_abs_val={max_abs_val:.4g}")

    return normalized

def calculate_information_deficits(normalized_features: Dict[str, np.ndarray],
                                   config: SEFAConfig) -> Dict[str, float]:
    """Calculates the information deficit w_X = Max(0, Log(B) - I_X)
       for each normalized feature.

    Implements SEFA.md Section II.5.b.
    Uses the global distribution of each normalized feature.
    Binning for the global distribution uses the configured entropy_binning method.

    Args:
        normalized_features (Dict[str, np.ndarray]): Dictionary of normalized features.
        config (SEFAConfig): Configuration object with entropy binning settings.

    Returns:
        Dict[str, float]: Dictionary of information deficits {w_A, w_C, w_F, w_E}.
    """
    deficits = {}
    for name, data in normalized_features.items():
        # Remove NaNs for global entropy calculation
        valid_data = data[~np.isnan(data)]
        if valid_data.size < 2:
            logger.warning(f"Feature '{name}' has < 2 valid points. Setting deficit to 0.")
            deficits[name] = 0.0
            continue

        # Determine bins for global distribution using configured method
        bin_method = config.entropy_binning
        bins: Union[int, str, np.ndarray] = 'auto'
        num_bins_used = 0

        if bin_method == 'knuth':
            try:
                bin_width, bins = knuth_bin_width(valid_data, return_bins=True)
                num_bins_used = len(bins) - 1 if len(bins) > 1 else 0
                if num_bins_used <= 0:
                     logger.debug(f"Knuth rule resulted in <= 0 bins for global feature '{name}'. Using fixed bins={config.entropy_bin_count} as fallback.")
                     bins = config.entropy_bin_count
                     num_bins_used = config.entropy_bin_count
            except Exception as e:
                logger.warning(f"Knuth bin calculation failed for global feature '{name}': {e}. Falling back to fixed bins ({config.entropy_bin_count}).")
                bins = config.entropy_bin_count
                num_bins_used = config.entropy_bin_count
        elif bin_method == 'fixed':
            bins = config.entropy_bin_count
            num_bins_used = config.entropy_bin_count
        else:
            raise ValueError(f"Unknown entropy_binning method: {bin_method}")

        if num_bins_used <= 0:
             logger.error(f"Number of bins used for global feature '{name}' is {num_bins_used}. Cannot calculate entropy. Setting deficit to 0.")
             deficits[name] = 0.0
             continue

        # Calculate histogram and entropy (I_X)
        try:
            counts, bin_edges = np.histogram(valid_data, bins=bins)
            counts = counts[counts > 0]
            if counts.size == 0:
                 feature_entropy_Ix = 0.0 # Zero entropy if all data in one bin
            else:
                 feature_entropy_Ix = scipy_entropy(counts, base=np.e)
        except ValueError as ve:
            logger.error(f"Error during histogram calculation for global feature '{name}': {ve}. Setting deficit to 0.")
            deficits[name] = 0.0
            continue

        # Calculate theoretical maximum entropy (Log(B)) using actual number of bins used
        # Note: SEFA.md uses Log(B), where B is often fixed. Here we use the *actual*
        # number of bins determined by Knuth or fixed setting for consistency.
        max_entropy_logB = np.log(num_bins_used)

        # Calculate deficit w_X = Max(0, Log(B) - I_X)
        deficit = max(0.0, max_entropy_logB - feature_entropy_Ix)
        deficits[name] = deficit
        logger.debug(f"Deficit for '{name}': max_logB={max_entropy_logB:.4f}, Ix={feature_entropy_Ix:.4f}, w_X={deficit:.4f}")

    return deficits

def calculate_exponents(information_deficits: Dict[str, float], p: int) -> Dict[str, float]:
    """Calculates the exponents alpha_X = p * w_X / W_Total.

    Implements SEFA.md Section II.5.b, Equation (22).

    Args:
        information_deficits (Dict[str, float]): Dictionary of feature deficits {w_X}.
        p (int): Number of features being combined (scaling factor).

    Returns:
        Dict[str, float]: Dictionary of calculated exponents {alpha_X}.
    """
    W_Total = sum(information_deficits.values())
    exponents = {}

    if W_Total <= 0:
        # Avoid division by zero. Assign equal weights? Or zero?
        # If W_Total is 0, all deficits were 0, meaning all features had max entropy.
        # Assigning equal weights seems reasonable in this degenerate case.
        warnings.warn(f"Total information deficit W_Total is {W_Total:.4g}. Assigning equal exponents.", UserWarning)
        num_features = len(information_deficits)
        equal_exponent = p / num_features if num_features > 0 else 0
        for name in information_deficits:
            exponents[name] = equal_exponent
    else:
        for name, w_X in information_deficits.items():
            alpha_X = p * w_X / W_Total
            exponents[name] = alpha_X

    # Verify sum of exponents equals p (within tolerance)
    sum_alpha = sum(exponents.values())
    if not np.isclose(sum_alpha, p):
        logger.warning(f"Sum of exponents ({sum_alpha:.4f}) does not equal p ({p}). Check deficit calculations.")

    return exponents

def calculate_sefa_score(normalized_features: Dict[str, np.ndarray],
                         exponents_alpha: Dict[str, float],
                         epsilon: float) -> np.ndarray:
    """Calculates the final SEFA score using the log-domain formulation.

    Implements SEFA.md Section II.6, Equation (23) and the log-domain version.
    LogSEFA(y) = Sum[ alpha_X * Log(| X_tilde(y) | ) ]
    SEFA(y) = Exp( LogSEFA(y) )
    where X_tilde(y) = Sign(X'(y)) * Max(epsilon, |X'(y)|)

    Args:
        normalized_features (Dict[str, np.ndarray]): Dictionary of normalized features {X'}.
        exponents_alpha (Dict[str, float]): Dictionary of calculated exponents {alpha_X}.
        epsilon (float): Small regularization constant.

    Returns:
        np.ndarray: The final SEFA score array.
    """
    log_sefa_terms = []
    feature_names = list(normalized_features.keys())

    # Get the length from the first valid feature array
    array_len = 0
    for name in feature_names:
        if normalized_features[name] is not None and normalized_features[name].ndim > 0:
             array_len = len(normalized_features[name])
             break
    if array_len == 0:
         logger.error("Cannot determine array length from features. Returning empty array.")
         return np.array([])

    total_log_sefa = np.zeros(array_len, dtype=float)

    for name in feature_names:
        if name not in exponents_alpha:
            logger.warning(f"Exponent alpha not found for feature '{name}'. Skipping its contribution.")
            continue

        alpha_X = exponents_alpha[name]
        X_prime = normalized_features[name]

        if alpha_X == 0:
            continue # Skip features with zero weight

        # Handle potential NaNs in input features
        nan_mask = np.isnan(X_prime)
        valid_X_prime = X_prime[~nan_mask]

        # Calculate X_tilde = Sign(X') * Max(epsilon, |X'|)
        # We only need the magnitude |X_tilde| = Max(epsilon, |X'|) for the log
        magnitude_X_tilde = np.maximum(epsilon, np.abs(valid_X_prime))

        # Calculate Log(|X_tilde|) - use natural log (ln)
        log_magnitude_X_tilde = np.log(magnitude_X_tilde)

        # Calculate term: alpha_X * Log(|X_tilde|)
        term = alpha_X * log_magnitude_X_tilde

        # Add term to total, placing NaNs back
        term_full = np.full(len(X_prime), np.nan)
        term_full[~nan_mask] = term
        total_log_sefa = np.nansum([total_log_sefa, term_full], axis=0)
        # Note: nansum treats NaNs as zero, which is correct here since
        # if a feature is NaN, its contribution to the sum should be zero-like,
        # and the final score at that point will likely be NaN or zero after exp.

    # Final SEFA score: Exp(LogSEFA)
    # Handle potential underflow/overflow with exp
    # Clip LogSEFA to avoid extreme values? e.g., np.clip(total_log_sefa, -700, 700)
    # For now, let exp handle it.
    sefa_score = np.exp(total_log_sefa)

    # Ensure final score is non-negative and finite, replace NaNs from intermediate steps
    sefa_score = np.nan_to_num(sefa_score, nan=0.0, posinf=np.nanmax(sefa_score[np.isfinite(sefa_score)]), neginf=0.0)
    sefa_score = np.maximum(0, sefa_score)

    return sefa_score 