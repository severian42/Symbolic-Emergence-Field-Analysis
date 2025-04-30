"""Configuration constants and default parameters for SEFA."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union

# Default value for the small positive regularization constant (Section II.6)
DEFAULT_EPSILON = 1e-16

# Default weight decay exponent (beta in w[k] = 1/(1 + gamma[k]^beta)) (Section II.2, Appendix B update)
DEFAULT_BETA = 2.0

# Default number of features combined in the SEFA score (p) (Section II.5.b)
DEFAULT_P_FEATURES = 4

# Default method for calculating derivatives (Frequency and Curvature)
# 'finite_difference': Simple but potentially noisy (Not recommended per Limitation 2.3)
# 'savgol': Savitzky-Golay filter before differentiation (Recommended per Limitation 2.3)
# 'polyfit': Local polynomial fitting before differentiation (Recommended per Limitation 2.3)
DEFAULT_DERIVATIVE_METHOD: Literal['finite_difference', 'savgol', 'polyfit'] = 'savgol'

# Default method for entropy binning
# 'knuth': Knuth's rule for adaptive binning (Recommended per Limitation 2.2)
# 'fixed': Fixed number of bins (Simpler, use bin_count)
DEFAULT_ENTROPY_BINNING: Literal['knuth', 'fixed'] = 'knuth'

# Default number of bins if using 'fixed' entropy binning
DEFAULT_ENTROPY_BIN_COUNT = 32

# Default method for handling Hilbert transform boundaries (Limitation 2.1)
# 'discard': Discard a fraction of the boundaries
# 'mirror': Use mirror padding before FFT
# 'periodic': Use standard FFT assumption (potential artifacts)
DEFAULT_BOUNDARY_METHOD: Literal['discard', 'mirror', 'periodic'] = 'discard'

# Fraction of the domain to discard at each end if boundary_method is 'discard'
DEFAULT_BOUNDARY_DISCARD_FRACTION = 0.05 # Discard 5% from each end


@dataclass
class SavgolParams:
    """Parameters for Savitzky-Golay filtering used in derivative calculation."""
    window_length: int = 51 # Must be odd
    polyorder: int = 3
    deriv_order: int = 1 # 1 for first derivative (Frequency), 2 for second (Curvature)

@dataclass
class PolyfitParams:
    """Parameters for local polynomial fitting used in derivative calculation."""
    window_length: int = 51 # Must be odd
    polyorder: int = 3
    deriv_order: int = 1 # 1 for first derivative (Frequency), 2 for second (Curvature)


@dataclass
class SEFAConfig:
    """
    Configuration parameters for the SEFA algorithm.

    Defaults are based on the recommendations and definitions in SEFA.md.

    Attributes:
        epsilon: Small positive regularization constant for numerical stability in logarithms.
                 (Default: 1e-16, see SEFA.md Section II.6).
        beta: Exponent for driver weight calculation (w_k = 1 / (1 + gamma_k^beta)).
              (Default: 2.0, see SEFA.md Section II.2, Appendix B update).
        p_features: Number of features combined in the final SEFA score (scaling factor p).
                    (Default: 4 for A, C, F, E, see SEFA.md Section II.5.b).
        derivative_method: Method used for calculating instantaneous frequency and curvature.
                           Options: 'finite_difference', 'savgol', 'polyfit'.
                           ('savgol' or 'polyfit' recommended, see SEFA.md Limitation 2.3).
                           (Default: 'savgol').
        savgol_frequency_params: Parameters for Savitzky-Golay if used for frequency (1st derivative).
                                 (Default: window=51, polyorder=3, deriv=1).
        savgol_curvature_params: Parameters for Savitzky-Golay if used for curvature (2nd derivative).
                                 (Default: window=51, polyorder=3, deriv=2).
        polyfit_frequency_params: Parameters for polynomial fitting if used for frequency (1st derivative).
                                 (Default: window=51, polyorder=3, deriv=1).
        polyfit_curvature_params: Parameters for polynomial fitting if used for curvature (2nd derivative).
                                 (Default: window=51, polyorder=3, deriv=2).
        entropy_window_size: Size of the sliding window for local entropy calculation.
                             Needs optimization (SEFA.md Section II.4.a, Limitation 2.2).
                             Required, no default.
        entropy_binning: Method for determining bins for entropy calculation ('knuth' or 'fixed').
                         ('knuth' recommended, see SEFA.md Limitation 2.2).
                         (Default: 'knuth').
        entropy_bin_count: Number of bins if entropy_binning is 'fixed'.
                           (Default: 32).
        # TODO (Limitation 2.2): Add support for alternative entropy estimators for small windows (W < 50),
        # e.g., 'knn' or 'kl'. Currently uses histogram-based methods.
        small_window_entropy_method: Optional[Literal['knn', 'kl']] = None
        boundary_method: Method for handling Hilbert transform edge effects.
                         Options: 'discard', 'mirror', 'periodic'.
                         ('discard' or 'mirror' recommended, see SEFA.md Limitation 2.1).
                         (Default: 'discard').
        boundary_discard_fraction: Fraction of data to discard from each end if boundary_method='discard'.
                                   (Default: 0.05).
        # TODO (Limitation 1.3): Add option for tunable weight function exponent (beta) optimization.
        # TODO (Limitation 2.1.1): Add option for tunable weight function or windowing for drivers.
    """
    epsilon: float = DEFAULT_EPSILON
    beta: float = DEFAULT_BETA
    p_features: int = DEFAULT_P_FEATURES
    derivative_method: Literal['finite_difference', 'savgol', 'polyfit'] = DEFAULT_DERIVATIVE_METHOD
    savgol_frequency_params: SavgolParams = field(default_factory=lambda: SavgolParams(deriv_order=1))
    savgol_curvature_params: SavgolParams = field(default_factory=lambda: SavgolParams(deriv_order=2))
    polyfit_frequency_params: PolyfitParams = field(default_factory=lambda: PolyfitParams(deriv_order=1))
    polyfit_curvature_params: PolyfitParams = field(default_factory=lambda: PolyfitParams(deriv_order=2))
    entropy_window_size: int = field(default=None) # Must be provided
    entropy_binning: Literal['knuth', 'fixed'] = DEFAULT_ENTROPY_BINNING
    entropy_bin_count: int = DEFAULT_ENTROPY_BIN_COUNT
    boundary_method: Literal['discard', 'mirror', 'periodic'] = DEFAULT_BOUNDARY_METHOD
    boundary_discard_fraction: float = DEFAULT_BOUNDARY_DISCARD_FRACTION

    def __post_init__(self):
        if self.entropy_window_size is None:
            raise ValueError("entropy_window_size must be specified in SEFAConfig.")
        if not isinstance(self.entropy_window_size, int) or self.entropy_window_size <= 0:
             raise ValueError("entropy_window_size must be a positive integer.")
        if self.derivative_method not in ['finite_difference', 'savgol', 'polyfit']:
            raise ValueError("derivative_method must be 'finite_difference', 'savgol', or 'polyfit'.")
        if self.entropy_binning not in ['knuth', 'fixed']:
            raise ValueError("entropy_binning must be 'knuth' or 'fixed'.")
        if self.boundary_method not in ['discard', 'mirror', 'periodic']:
             raise ValueError("boundary_method must be 'discard', 'mirror', or 'periodic'.")
        if self.boundary_method == 'discard' and not (0 < self.boundary_discard_fraction < 0.5):
             raise ValueError("boundary_discard_fraction must be between 0 and 0.5.")
        # TODO: Add validation for Savgol/Polyfit window lengths (must be odd, <= data length etc.)

# Global default configuration instance
DEFAULT_CONFIG = SEFAConfig(entropy_window_size=51) # Example window size, should be optimized

# Example of creating a custom configuration:
# custom_config = SEFAConfig(
#     epsilon=1e-15,
#     beta=2.5,
#     derivative_method='polyfit',
#     entropy_window_size=101,
#     entropy_binning='knuth',
#     boundary_method='mirror'
# ) 