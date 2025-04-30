"""Core SEFA class definition implementing the pipeline from SEFA.md."""

import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple

from .config import SEFAConfig, DEFAULT_CONFIG
from .domain import discretize_domain
from .drivers import calculate_driver_weights
from .transform import construct_field, get_analytic_signal
from .features import (
    extract_amplitude,
    extract_phase,
    extract_frequency,
    extract_curvature
)
from .entropy import calculate_sliding_window_entropy, calculate_entropy_alignment
from .scoring import (
    normalize_features,
    calculate_information_deficits,
    calculate_exponents,
    calculate_sefa_score
)
from .thresholding import apply_threshold
from .utils import handle_boundaries

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SEFA:
    """Main class for performing Symbolic Emergence Field Analysis (SEFA).

    This class implements the SEFA pipeline as described in SEFA.md,
    integrating steps from domain discretization to final score calculation
    and thresholding.

    Attributes:
        config (SEFAConfig): Configuration parameters for the analysis.
        domain_y (np.ndarray): Discretized domain points.
        dy (float): Step size in the discretized domain.
        drivers_gamma (np.ndarray): Input driver parameters (e.g., frequencies).
        weights_w (np.ndarray): Calculated driver weights.
        field_V0 (np.ndarray): Constructed real-valued field.
        analytic_signal_full (np.ndarray): Complex analytic signal (V0 + i*H(V0)) before boundary handling.
        original_indices (Optional[np.ndarray]): Indices mapping processed data back to original domain size if boundaries were discarded.
        analytic_signal (np.ndarray): Complex analytic signal (V0 + i*H(V0)) after boundary handling.
        processed_domain_y (np.ndarray): Domain corresponding to processed signal.
        processed_indices (Optional[np.ndarray]): Actual indices kept after processing.
        amplitude_A (np.ndarray): Envelope amplitude |AnalyticSignal|.
        phase_phi (np.ndarray): Unwrapped instantaneous phase Arg(AnalyticSignal).
        frequency_F (np.ndarray): Instantaneous frequency (d(phi)/dy).
        curvature_C (np.ndarray): Curvature of amplitude (d^2(A)/dy^2).
        entropy_S (np.ndarray): Local sliding window entropy.
        entropy_alignment_E (np.ndarray): Entropy alignment score (1 - S/SMax).
        features (Dict[str, np.ndarray]): Dictionary holding the calculated features {A, C, F, E}.
        normalized_features (Dict[str, np.ndarray]): Dictionary holding normalized features {A', C', F', E'}.
        information_deficits_w (Dict[str, float]): Information deficits {w_A, w_C, w_F, w_E}.
        exponents_alpha (Dict[str, float]): Calculated exponents {alpha_A, alpha_C, alpha_F, alpha_E}.
        sefa_score (np.ndarray): Final composite SEFA score.
        thresholded_mask (Optional[np.ndarray]): Boolean mask indicating regions above the threshold.
    """

    def __init__(self, config: SEFAConfig = DEFAULT_CONFIG):
        """Initializes the SEFA analyzer with a given configuration."""
        self.config = config
        logger.info(f"Initializing SEFA with config: {config}")

        # Initialize results containers
        self.domain_y: Optional[np.ndarray] = None
        self.dy: Optional[float] = None
        self.drivers_gamma: Optional[np.ndarray] = None
        self.weights_w: Optional[np.ndarray] = None
        self.field_V0: Optional[np.ndarray] = None
        self.analytic_signal_full: Optional[np.ndarray] = None
        self.original_indices: Optional[np.ndarray] = None
        self.analytic_signal: Optional[np.ndarray] = None
        self.processed_domain_y: Optional[np.ndarray] = None
        self.processed_indices: Optional[np.ndarray] = None
        self.amplitude_A: Optional[np.ndarray] = None
        self.phase_phi: Optional[np.ndarray] = None
        self.frequency_F: Optional[np.ndarray] = None
        self.curvature_C: Optional[np.ndarray] = None
        self.entropy_S: Optional[np.ndarray] = None
        self.entropy_alignment_E: Optional[np.ndarray] = None
        self.features: Dict[str, np.ndarray] = {}
        self.normalized_features: Dict[str, np.ndarray] = {}
        self.information_deficits_w: Dict[str, float] = {}
        self.exponents_alpha: Dict[str, float] = {}
        self.sefa_score: Optional[np.ndarray] = None
        self.thresholded_mask: Optional[np.ndarray] = None

    def run_pipeline(self,
                     drivers_gamma: np.ndarray,
                     ymin: float,
                     ymax: float,
                     num_points: int,
                     feature_list: List[str] = ['A', 'C', 'F', 'E']) -> np.ndarray:
        """Executes the full SEFA pipeline.

        Args:
            drivers_gamma (np.ndarray): Array of driver parameters (gamma_k).
            ymin (float): Minimum value of the domain.
            ymax (float): Maximum value of the domain.
            num_points (int): Number of points (M) to discretize the domain into.
            feature_list (List[str]): List of features to compute and combine.
                                      Defaults to ['A', 'C', 'F', 'E'].
                                      Must match config.p_features if using default.

        Returns:
            np.ndarray: The calculated SEFA score.
        """
        logger.info("Starting SEFA pipeline...")
        self.drivers_gamma = np.asarray(drivers_gamma)

        if len(feature_list) != self.config.p_features:
            logger.warning(f"Number of features in feature_list ({len(feature_list)}) "
                           f"does not match config.p_features ({self.config.p_features}). "
                           f"Using {len(feature_list)} for exponent calculation.")
            # Dynamically adjust p if needed, or raise error?
            # For now, let scoring use len(feature_list)
            effective_p = len(feature_list)
        else:
            effective_p = self.config.p_features

        # 1. Discretize domain (SEFA.md Section II.1)
        self.domain_y, self.dy = discretize_domain(ymin, ymax, num_points)
        logger.info(f"Domain discretized: [{ymin}, {ymax}] into {num_points} points, dy={self.dy:.4g}")

        # 2. Compute weights (SEFA.md Section II.2)
        self.weights_w = calculate_driver_weights(self.drivers_gamma, self.config.beta)
        logger.info(f"Calculated {len(self.weights_w)} driver weights using beta={self.config.beta}")

        # 3. Construct field (SEFA.md Section II.2)
        self.field_V0 = construct_field(self.domain_y, self.drivers_gamma, self.weights_w)
        logger.info("Constructed base field V0")

        # 4. Hilbert transform & Analytic signal (SEFA.md Section II.3.1, II.3.2)
        # Boundary handling is applied here (Limitation 2.1)
        self.analytic_signal_full, self.original_indices = get_analytic_signal(self.field_V0, self.config)
        # Apply boundary handling to the signal
        self.analytic_signal = handle_boundaries(self.analytic_signal_full, self.config, self.original_indices)

        # Determine the actual indices and corresponding domain *after* boundary handling
        if self.config.boundary_method == 'discard':
            n_original = len(self.domain_y)
            discard_count = int(n_original * self.config.boundary_discard_fraction)
            if discard_count > 0:
                start_idx = discard_count
                end_idx = n_original - discard_count
                self.processed_indices = np.arange(start_idx, end_idx)
                self.processed_domain_y = self.domain_y[self.processed_indices]
                logger.info(f"Discarding boundaries, {len(self.processed_domain_y)} points remaining.")
            else:
                # No discarding occurred
                self.processed_indices = np.arange(n_original)
                self.processed_domain_y = self.domain_y
        elif self.config.boundary_method == 'mirror':
            # For mirror padding, original_indices contains the slice needed
            if self.original_indices is not None:
                self.processed_indices = self.original_indices # These indices apply to the *original* domain
                self.processed_domain_y = self.domain_y[self.processed_indices]
            else:
                # Should not happen if padding was done correctly
                self.processed_indices = np.arange(len(self.domain_y))
                self.processed_domain_y = self.domain_y
        else: # periodic or other
            self.processed_indices = np.arange(len(self.domain_y))
            self.processed_domain_y = self.domain_y

        logger.info("Calculated analytic signal and handled boundaries.")
        # Ensure processed domain and signal lengths match *before* feature extraction
        assert len(self.processed_domain_y) == len(self.analytic_signal), \
            f"Internal shape mismatch: processed_domain_y ({len(self.processed_domain_y)}) vs analytic_signal ({len(self.analytic_signal)})"

        # 5-11. Feature Extraction (use the correctly processed domain and signal)
        logger.info("Extracting features...")
        # Pass the correctly sliced domain to feature extraction
        results = self._extract_all_features(self.processed_domain_y, self.analytic_signal, self.dy)
        self.amplitude_A = results['A']
        self.phase_phi = results['phi'] # Keep for potential diagnostics
        self.frequency_F = results['F']
        self.curvature_C = results['C']
        self.entropy_S = results['S']
        self.entropy_alignment_E = results['E']

        self.features = {key: val for key, val in results.items() if key in feature_list}
        logger.info(f"Extracted features: {list(self.features.keys())}")

        # 12. Normalize features (SEFA.md Section II.5.a)
        self.normalized_features = normalize_features(self.features)
        logger.info("Normalized features")

        # 13. Calculate Information Deficits (SEFA.md Section II.5.b)
        self.information_deficits_w = calculate_information_deficits(self.normalized_features, self.config)
        logger.info(f"Calculated information deficits: {self.information_deficits_w}")

        # 14. Calculate Exponents (SEFA.md Section II.5.b)
        self.exponents_alpha = calculate_exponents(self.information_deficits_w, p=effective_p)
        logger.info(f"Calculated exponents: {self.exponents_alpha}")

        # 15-16. Calculate SEFA Score (SEFA.md Section II.6)
        self.sefa_score = calculate_sefa_score(self.normalized_features, self.exponents_alpha, self.config.epsilon)
        logger.info("Calculated final SEFA score")

        logger.info("SEFA pipeline finished.")
        return self.sefa_score

    def _extract_all_features(self, domain_y: np.ndarray, analytic_signal: np.ndarray, dy: float) -> Dict[str, np.ndarray]:
        """Helper method to calculate all primary features."""
        results = {}

        # 6. Envelope Amplitude (SEFA.md Section II.3.3)
        results['A'] = extract_amplitude(analytic_signal)
        logger.debug("Extracted Amplitude (A)")

        # 7. Phase (SEFA.md Section II.3.4) - unwrapped
        # Phase itself isn't directly used in score but needed for frequency
        results['phi'] = extract_phase(analytic_signal, unwrap=True)
        logger.debug("Extracted and unwrapped Phase (phi)")

        # 8. Frequency (SEFA.md Section II.3.5)
        results['F'] = extract_frequency(results['phi'], dy, self.config)
        logger.debug(f"Extracted Frequency (F) using method: {self.config.derivative_method}")

        # 9. Curvature (SEFA.md Section II.3.6)
        results['C'] = extract_curvature(results['A'], dy, self.config)
        logger.debug(f"Extracted Curvature (C) using method: {self.config.derivative_method}")

        # 10. Entropy (SEFA.md Section II.4.a)
        results['S'] = calculate_sliding_window_entropy(results['A'], self.config)
        logger.debug(f"Calculated sliding window Entropy (S) using window={self.config.entropy_window_size}, binning={self.config.entropy_binning}")

        # 11. Entropy Alignment (SEFA.md Section II.4.b)
        results['E'] = calculate_entropy_alignment(results['S'])
        logger.debug("Calculated Entropy Alignment (E)")

        return results

    def threshold_score(self, method: str = 'otsu', **kwargs) -> Optional[np.ndarray]:
        """Applies thresholding to the calculated SEFA score.

        Args:
            method (str): Thresholding method ('otsu', 'percentile', 'mog').
                          See thresholding.py for details.
                          (Default: 'otsu').
            **kwargs: Additional arguments passed to the thresholding function.

        Returns:
            Optional[np.ndarray]: A boolean mask indicating regions above the threshold,
                                  or None if the score hasn't been calculated yet.
        """
        if self.sefa_score is None:
            logger.error("SEFA score has not been calculated yet. Run run_pipeline() first.")
            return None

        logger.info(f"Applying thresholding method: {method}")
        self.thresholded_mask = apply_threshold(self.sefa_score, method=method, **kwargs)
        logger.info(f"Thresholding complete. {np.sum(self.thresholded_mask)} points above threshold.")
        return self.thresholded_mask

    def get_results(self) -> Dict[str, Any]:
        """Returns a dictionary containing all calculated results."""
        return {
            "config": self.config,
            "domain_y": self.domain_y, # Original full domain
            # Return the correctly processed domain stored during pipeline run
            "processed_domain_y": self.processed_domain_y,
            "dy": self.dy,
            "drivers_gamma": self.drivers_gamma,
            "weights_w": self.weights_w,
            "field_V0": self.field_V0, # Original field before boundary handling
            "analytic_signal": self.analytic_signal, # Processed signal
            "amplitude_A": self.amplitude_A,
            "phase_phi": self.phase_phi,
            "frequency_F": self.frequency_F,
            "curvature_C": self.curvature_C,
            "entropy_S": self.entropy_S,
            "entropy_alignment_E": self.entropy_alignment_E,
            "features": self.features,
            "normalized_features": self.normalized_features,
            "information_deficits_w": self.information_deficits_w,
            "exponents_alpha": self.exponents_alpha,
            "sefa_score": self.sefa_score,
            "thresholded_mask": self.thresholded_mask,
            # Return the indices used for processing (might differ from original_indices if discard)
            "processed_indices": self.processed_indices,
            # original_indices is now more specifically about padding mapping if mirror was used
            "original_indices_padding_map": self.original_indices if self.config.boundary_method == 'mirror' else None
        }

# Example usage:
# if __name__ == '__main__':
#     # Load drivers (e.g., zeta zeros)
#     drivers = np.loadtxt('path/to/zetazeros-50k.txt') # Hypothetical path
#
#     # Configure SEFA (using defaults or customizing)
#     config = SEFAConfig(
#         entropy_window_size=101, # Example: Needs optimization
#         boundary_method='discard',
#         boundary_discard_fraction=0.05,
#         derivative_method='savgol'
#     )
#
#     analyzer = SEFA(config=config)
#
#     # Define domain (e.g., log domain for integers)
#     N_min, N_max = 2, 1000
#     y_min, y_max = np.log(N_min), np.log(N_max)
#     M = 50000 # Number of points
#
#     # Run the pipeline
#     sefa_score = analyzer.run_pipeline(drivers, y_min, y_max, M)
#
#     # Get processed domain corresponding to the score
#     results = analyzer.get_results()
#     processed_y = results['processed_domain_y']
#     original_indices = results['original_indices'] # Use this to map back if needed
#
#     # Apply thresholding
#     mask = analyzer.threshold_score(method='otsu')
#
#     # Analyze results (e.g., plot score, find peaks in masked regions)
#     if processed_y is not None and sefa_score is not None:
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(12, 6))
#         plt.plot(processed_y, sefa_score, label='SEFA Score')
#         if mask is not None:
#             threshold_val = # ... (need to get threshold value from apply_threshold or estimate)
#             # plt.hlines(threshold_val, processed_y[0], processed_y[-1], color='r', linestyle='--', label=f'Otsu Threshold')
#             plt.fill_between(processed_y, 0, sefa_score, where=mask, alpha=0.3, color='orange', label='Above Threshold')
#         plt.xlabel('Domain y (e.g., log(N))')
#         plt.ylabel('SEFA Score')
#         plt.title('SEFA Analysis')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#
#         # Map peaks back to original domain (e.g., integer N)
#         if mask is not None:
#             peak_indices_processed = np.where(mask)[0] # Simple example, use proper peak finding
#             if original_indices is not None:
#                  peak_indices_original = original_indices[peak_indices_processed]
#                  peak_y_values = analyzer.domain_y[peak_indices_original]
#                  peak_N_values = np.exp(peak_y_values)
#                  print("Detected symbolic points (N approx.):", np.round(peak_N_values).astype(int)) 