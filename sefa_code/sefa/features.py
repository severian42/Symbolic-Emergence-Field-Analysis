"""Functions for extracting geometric features from the analytic signal."""

import numpy as np
import logging

from .config import SEFAConfig
from .utils import calculate_derivative

logger = logging.getLogger(__name__)

def extract_amplitude(analytic_signal: np.ndarray) -> np.ndarray:
    """Calculates the envelope amplitude A(y) = |AnalyticSignal(y)|.

    Implements SEFA.md Section II.3.3.

    Args:
        analytic_signal (np.ndarray): The complex analytic signal.

    Returns:
        np.ndarray: The envelope amplitude.
    """
    amplitude = np.abs(analytic_signal)
    # Check for NaNs or Infs that might arise from Hilbert transform issues
    if np.any(~np.isfinite(amplitude)):
        # Propagate NaNs/Infs, don't convert them.
        logger.warning("Non-finite values found in amplitude. Check Hilbert transform output.")
        # amplitude = np.nan_to_num(amplitude) # Removed: Let NaNs propagate
    return amplitude

def extract_phase(analytic_signal: np.ndarray, unwrap: bool = True) -> np.ndarray:
    """Calculates the instantaneous phase phi(y) = Arg(AnalyticSignal(y)).

    Implements SEFA.md Section II.3.4.

    Args:
        analytic_signal (np.ndarray): The complex analytic signal.
        unwrap (bool): Whether to unwrap the phase to maintain continuity.
                       Defaults to True, as required for frequency calculation.

    Returns:
        np.ndarray: The instantaneous phase (potentially unwrapped).
    """
    phase = np.angle(analytic_signal)
    if unwrap:
        # SEFA.md mentions phase unwrapping (e.g., Ghiglia & Pritt).
        # np.unwrap performs 1D unwrapping.
        # TODO: Consider more advanced unwrapping if np.unwrap fails on complex data.
        phase = np.unwrap(phase)
        logger.debug("Phase unwrapped using np.unwrap.")
    return phase

def extract_frequency(phase_phi: np.ndarray, dy: float, config: SEFAConfig) -> np.ndarray:
    """Calculates the instantaneous frequency F(y) = d(phi)/dy.

    Implements SEFA.md Section II.3.5.
    Uses the derivative calculation method specified in the config.

    Args:
        phase_phi (np.ndarray): The unwrapped instantaneous phase.
        dy (float): The domain step size.
        config (SEFAConfig): Configuration object with derivative settings.

    Returns:
        np.ndarray: The instantaneous frequency.
    """
    frequency = calculate_derivative(phase_phi, dy, config, deriv_order=1)
    return frequency

def extract_curvature(amplitude_A: np.ndarray, dy: float, config: SEFAConfig) -> np.ndarray:
    """Calculates the curvature of the amplitude C(y) = d^2(A)/dy^2.

    Implements SEFA.md Section II.3.6.
    Uses the derivative calculation method specified in the config.
    Note: SEFA.md recommends smoothing A(y) before differentiation; this is
    handled within the calculate_derivative function if using 'savgol' or 'polyfit'.

    Args:
        amplitude_A (np.ndarray): The envelope amplitude.
        dy (float): The domain step size.
        config (SEFAConfig): Configuration object with derivative settings.

    Returns:
        np.ndarray: The curvature of the amplitude.
    """
    curvature = calculate_derivative(amplitude_A, dy, config, deriv_order=2)
    return curvature

# TODO: Implement functions as per SEFA.md Section II.3.3 - II.3.6
# TODO: Address Limitation 2.3 (Derivative Noise) - smoothing/polynomial fit
# TODO: Address Limitation 1.2 / Update (Sign Preservation)
# TODO: Consider phase unwrapping from utils 

# TODO (np.unwrap): Consider more advanced unwrapping if np.unwrap fails on complex data (SEFA.md Sec II.3.4). 