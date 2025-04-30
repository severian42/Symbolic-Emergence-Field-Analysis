"""Functions for thresholding the SEFA score."""

import numpy as np
import logging
from typing import Optional
import warnings

# Attempt to import optional dependencies
try:
    from skimage.filters import threshold_otsu
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    threshold_otsu = None

try:
    from sklearn.mixture import GaussianMixture
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    GaussianMixture = None

logger = logging.getLogger(__name__)

def apply_threshold(sefa_score: np.ndarray, method: str = 'otsu', **kwargs) -> Optional[np.ndarray]:
    """Applies a thresholding method to the SEFA score to get a binary mask.

    Implements SEFA.md Section II.7.
    Available methods depend on installed libraries (skimage, sklearn).

    Args:
        sefa_score (np.ndarray): The calculated SEFA score array.
        method (str): The thresholding method to use.
            Options:
            - 'otsu': Otsu's method (requires scikit-image).
            - 'percentile': Threshold based on a percentile value.
                            Requires kwarg `percentile` (e.g., percentile=95).
            - 'mog': Gaussian Mixture Model (requires scikit-learn).
                     Assumes bimodal distribution (signal vs background).
                     Requires kwarg `n_components` (usually 2).
                     TODO: Implement logic to select the higher-mean component.
            - 'manual': Apply a manual threshold value.
                        Requires kwarg `threshold_value`.
        **kwargs: Additional keyword arguments required by the specific method.

    Returns:
        Optional[np.ndarray]: A boolean mask where True indicates values above the
                              threshold. Returns None if the method fails or dependencies
                              are missing.
    """
    # Handle NaNs - thresholding methods usually don't handle them well.
    valid_score = sefa_score[~np.isnan(sefa_score)]
    if valid_score.size == 0:
        logger.warning("SEFA score contains only NaNs. Cannot apply threshold.")
        return np.zeros_like(sefa_score, dtype=bool) # Return all False mask

    threshold_value: Optional[float] = None

    if method == 'otsu':
        if not _HAS_SKIMAGE or threshold_otsu is None:
            logger.error("Otsu thresholding requires scikit-image to be installed.")
            return None
        try:
            # Handle potential errors if data is constant
            if np.all(valid_score == valid_score[0]):
                 warnings.warn("SEFA score is constant. Otsu threshold is undefined. Returning mask based on value.", UserWarning)
                 # Arbitrarily return all True if constant value > 0, else False
                 return np.where(~np.isnan(sefa_score), sefa_score > 0, False)
            threshold_value = threshold_otsu(valid_score)
            logger.info(f"Otsu threshold calculated: {threshold_value:.4g}")
        except Exception as e:
            logger.error(f"Otsu threshold calculation failed: {e}")
            return None

    elif method == 'percentile':
        if 'percentile' not in kwargs:
            logger.error("Percentile thresholding requires the 'percentile' keyword argument (0-100).")
            return None
        percentile = kwargs['percentile']
        if not (0 <= percentile <= 100):
             logger.error("Percentile must be between 0 and 100.")
             return None
        threshold_value = np.percentile(valid_score, percentile)
        logger.info(f"{percentile}th percentile threshold calculated: {threshold_value:.4g}")

    elif method == 'mog':
        if not _HAS_SKLEARN or GaussianMixture is None:
            logger.error("Gaussian Mixture Model thresholding requires scikit-learn to be installed.")
            return None
        n_components = kwargs.get('n_components', 2)
        if n_components != 2:
             logger.warning("Using MoG with n_components != 2. Threshold interpretation might be complex.")

        try:
            gmm = GaussianMixture(n_components=n_components, random_state=0)
            gmm.fit(valid_score.reshape(-1, 1))

            # Find the threshold between the two components
            # Often taken as the point where posterior probabilities are equal,
            # or related to the means and covariances.
            # Simple approach for 2 components: midpoint between means? Or intersection?
            # For now, let's find the intersection point, assuming equal priors and variances? (Not ideal)
            # A common heuristic: find the minimum between the two peaks in the fitted density.
            # Or, use the mean of the higher component as a threshold?
            # Let's calculate means and identify the component with the higher mean.
            means = gmm.means_.flatten()
            weights = gmm.weights_
            covariances = gmm.covariances_.flatten()

            idx_max_mean = np.argmax(means)
            idx_min_mean = np.argmin(means)

            # Simple threshold: Midpoint between means (weighted by stddev?)
            # threshold_value = (means[0] + means[1]) / 2
            # Or, find intersection of the two Gaussian PDFs (more complex)
            # Or, use a value related to the higher mean component
            threshold_value = means[idx_min_mean] + 1.0 * np.sqrt(covariances[idx_min_mean]) # Example heuristic
            # TODO: Implement a more robust method to find the optimal threshold between MoG components.
            logger.info(f"MoG means: {means}. Using heuristic threshold: {threshold_value:.4g}")

        except Exception as e:
            logger.error(f"Gaussian Mixture Model fitting/thresholding failed: {e}")
            return None

    elif method == 'manual':
        if 'threshold_value' not in kwargs:
             logger.error("Manual thresholding requires the 'threshold_value' keyword argument.")
             return None
        threshold_value = kwargs['threshold_value']
        logger.info(f"Using manual threshold: {threshold_value:.4g}")

    else:
        logger.error(f"Unknown thresholding method: {method}")
        return None

    if threshold_value is None:
         logger.error("Threshold value could not be determined.")
         return None

    # Apply threshold, preserving NaNs as False in the mask
    mask = np.where(~np.isnan(sefa_score), sefa_score > threshold_value, False)
    return mask

# TODO (MoG): Implement a more robust method to find the optimal threshold between MoG components (SEFA.md Sec II.7).
# TODO: Consider adding other threshold methods if needed.

# TODO: Implement functions as per SEFA.md Section II.7
# TODO: Implement Otsu's method
# TODO: Implement percentile thresholding
# TODO: Implement Gaussian Mixture Model thresholding
# TODO: Address limitations of Otsu noted in SEFA.md 