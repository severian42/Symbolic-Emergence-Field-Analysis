"""Tests for feature scoring functions."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

# Conditional import for astropy
try:
    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False

from sefa.scoring import (
    normalize_features,
    calculate_information_deficits,
    calculate_exponents,
    calculate_sefa_score
)
from sefa.config import SEFAConfig

# --- Fixtures ---
@pytest.fixture
def default_config():
    return SEFAConfig(entropy_window_size=51) # Need window size for validation

@pytest.fixture
def sample_features():
    """Provide sample raw features for testing."""
    return {
        'A': np.array([1.0, 2.0, 1.5, 0.5, np.nan]), # MaxAbs=2.0
        'C': np.array([-10.0, 0.0, 5.0, -2.0, 1.0]), # MaxAbs=10.0
        'F': np.array([100.0, 150.0, 120.0, 90.0, 110.0]), # MaxAbs=150.0
        'E': np.array([0.9, 0.1, 0.5, 0.8, 0.3]), # MaxAbs=0.9
    }

@pytest.fixture
def normalized_features_simple():
    """Simple normalized features for testing deficits/exponents."""
    return {
        'A': np.array([0.5, 1.0, 0.5, 0.5]), # Low entropy -> High deficit
        'C': np.array([-1.0, 1.0, -1.0, 1.0]), # High entropy -> Low deficit
    }

# --- Test Functions ---

def test_normalize_features(sample_features):
    """Test feature normalization including sign preservation and NaNs."""
    norm = normalize_features(sample_features)

    assert list(norm.keys()) == list(sample_features.keys())

    # Check A (MaxAbs=2.0)
    expected_A = np.array([1.0/2.0, 2.0/2.0, 1.5/2.0, 0.5/2.0, np.nan])
    assert_allclose(norm['A'], expected_A, equal_nan=True)
    assert np.nanmin(norm['A']) >= 0 # A should be non-negative

    # Check C (MaxAbs=10.0)
    expected_C = np.array([-10.0/10.0, 0.0/10.0, 5.0/10.0, -2.0/10.0, 1.0/10.0])
    assert_allclose(norm['C'], expected_C)

    # Check F (MaxAbs=150.0)
    expected_F = np.array([100.0/150.0, 150.0/150.0, 120.0/150.0, 90.0/150.0, 110.0/150.0])
    assert_allclose(norm['F'], expected_F)

    # Check E (MaxAbs=0.9)
    expected_E = np.array([0.9/0.9, 0.1/0.9, 0.5/0.9, 0.8/0.9, 0.3/0.9])
    assert_allclose(norm['E'], expected_E)
    assert np.nanmin(norm['E']) >= 0 # E should be non-negative

def test_normalize_features_zero_max():
    """Test normalization when max absolute value is zero."""
    features = {'A': np.array([0.0, 0.0, np.nan, 0.0])}
    norm = normalize_features(features)
    expected = np.array([0.0, 0.0, np.nan, 0.0])
    assert_allclose(norm['A'], expected, equal_nan=True)

def test_normalize_features_all_nan():
    """Test normalization when all values are NaN."""
    features = {'A': np.array([np.nan, np.nan, np.nan])}
    norm = normalize_features(features)
    assert np.all(np.isnan(norm['A']))

@pytest.mark.skipif(not _HAS_ASTROPY, reason="astropy not installed")
def test_calculate_information_deficits_knuth(normalized_features_simple, default_config):
    """Test information deficit calculation with Knuth bins."""
    config = default_config
    config.entropy_binning = 'knuth'

    deficits = calculate_information_deficits(normalized_features_simple, config)

    # Feature A: [0.5, 1.0, 0.5, 0.5] -> Likely 2 bins (0.5, 1.0), H ~ log(2)
    # Max entropy depends on bins found, let's assume 2: log(2)
    # Deficit_A = log(2) - H ~ 0 (High deficit expected but formula yields low if H is high)
    # Let's recalculate: Counts=[3, 1], p=[0.75, 0.25], H=-(0.75*log(0.75)+0.25*log(0.25))=0.562
    # If Knuth finds 2 bins: MaxH=log(2)=0.693. Deficit = 0.693 - 0.562 = 0.131

    # Feature C: [-1.0, 1.0, -1.0, 1.0] -> Likely 2 bins (-1, 1), H = log(2)
    # Max entropy depends on bins found, assume 2: log(2)
    # Deficit_C = log(2) - log(2) = 0

    assert 'A' in deficits
    assert 'C' in deficits
    # Exact value depends heavily on Knuth binning result, check relative values
    assert deficits['A'] > deficits['C'] # Expect A (more structure) > C (less structure)
    assert deficits['C'] >= 0 # Deficits should be non-negative

def test_calculate_information_deficits_fixed(normalized_features_simple, default_config):
    """Test information deficit calculation with fixed bins."""
    config = default_config
    config.entropy_binning = 'fixed'
    config.entropy_bin_count = 2 # Use 2 bins for simplicity

    deficits = calculate_information_deficits(normalized_features_simple, config)

    # Feature A: [0.5, 1.0, 0.5, 0.5]. Bins (e.g., [0.5, 0.75), [0.75, 1.0])
    # Counts: [3, 1]. H = 0.562. MaxH = log(2) = 0.693. Deficit = 0.131
    # Feature C: [-1.0, 1.0, -1.0, 1.0]. Bins (e.g., [-1, 0), [0, 1.0])
    # Counts: [2, 2]. H = log(2) = 0.693. MaxH = log(2) = 0.693. Deficit = 0

    assert_allclose(deficits['A'], 0.1308, atol=1e-4)
    assert_allclose(deficits['C'], 0.0, atol=1e-9)

def test_calculate_exponents():
    """Test exponent calculation alpha = p * w / W_total."""
    deficits = {'A': 0.6, 'C': 0.1, 'F': 0.3, 'E': 0.0} # W_total = 1.0
    p = 4
    exponents = calculate_exponents(deficits, p)
    expected_exponents = {
        'A': 4 * 0.6 / 1.0, # 2.4
        'C': 4 * 0.1 / 1.0, # 0.4
        'F': 4 * 0.3 / 1.0, # 1.2
        'E': 4 * 0.0 / 1.0, # 0.0
    }
    assert exponents.keys() == expected_exponents.keys()
    for key in exponents:
        assert_allclose(exponents[key], expected_exponents[key])
    # Check sum = p
    assert_allclose(sum(exponents.values()), p)

def test_calculate_exponents_zero_total_deficit():
    """Test exponent calculation when total deficit is zero (equal weights)."""
    deficits = {'A': 0.0, 'C': 0.0} # W_total = 0.0
    p = 2
    # Should warn and assign equal weights
    with pytest.warns(UserWarning, match="Assigning equal exponents"):
        exponents = calculate_exponents(deficits, p)
    expected_exponent = p / len(deficits) # 2 / 2 = 1.0
    assert exponents == {'A': expected_exponent, 'C': expected_exponent}

def test_calculate_sefa_score():
    """Test the final SEFA score calculation (log-domain)."""
    norm_features = {
        'A': np.array([0.1, 1.0, 0.5]),
        'C': np.array([-0.2, 0.8, -1.0])
    }
    exponents = {'A': 1.5, 'C': 0.5}
    epsilon = 1e-16

    # LogSEFA = alpha_A * log(|A_tilde|) + alpha_C * log(|C_tilde|)
    # Point 1: 1.5*log(0.1) + 0.5*log(0.2) = 1.5*(-2.302) + 0.5*(-1.609) = -3.453 - 0.805 = -4.258
    # Point 2: 1.5*log(1.0) + 0.5*log(0.8) = 1.5*(0) + 0.5*(-0.223) = -0.1115
    # Point 3: 1.5*log(0.5) + 0.5*log(1.0) = 1.5*(-0.693) + 0.5*(0) = -1.0395
    expected_log_sefa = np.array([-4.2579, -0.11157, -1.0397])
    expected_sefa = np.exp(expected_log_sefa) # [0.0141, 0.8944, 0.3535]

    sefa_score = calculate_sefa_score(norm_features, exponents, epsilon)
    assert_allclose(sefa_score, expected_sefa, rtol=1e-4, atol=1e-5)
    assert np.all(sefa_score >= 0)

def test_calculate_sefa_score_epsilon():
    """Test epsilon clipping in SEFA score calculation."""
    norm_features = {'A': np.array([0.0, 0.5])}
    exponents = {'A': 1.0}
    epsilon = 1e-3

    # Point 1: alpha_A * log(max(epsilon, |0.0|)) = 1.0 * log(1e-3) = -6.9077
    # Point 2: alpha_A * log(max(epsilon, |0.5|)) = 1.0 * log(0.5) = -0.6931
    expected_log_sefa = np.array([-6.9077, -0.6931])
    expected_sefa = np.exp(expected_log_sefa) # [0.001, 0.5]

    sefa_score = calculate_sefa_score(norm_features, exponents, epsilon)
    assert_allclose(sefa_score, expected_sefa, rtol=1e-4)

def test_calculate_sefa_score_nan():
    """Test SEFA score calculation with NaNs in features."""
    norm_features = {
        'A': np.array([0.1, np.nan, 0.5]),
        'C': np.array([-0.2, 0.8, np.nan])
    }
    exponents = {'A': 1.5, 'C': 0.5}
    epsilon = 1e-16

    sefa_score = calculate_sefa_score(norm_features, exponents, epsilon)

    # Point 1: 1.5*log(0.1) + 0.5*log(0.2) -> exp(-4.258) = 0.014
    # Point 2: A is NaN -> treated as 0 contribution. LogSEFA = 0.5*log(0.8) = -0.1115 -> exp = 0.894
    # Point 3: C is NaN -> treated as 0 contribution. LogSEFA = 1.5*log(0.5) = -1.0397 -> exp = 0.353
    # nan_to_num should handle NaNs in final output, making them 0.
    expected_sefa = np.array([0.01415, 0.89443, 0.35355]) # Based on log values above
    expected_sefa_nan_handling = np.array([0.01415, 0.89443, 0.35355]) # nansum treats NaN contribution as 0
                                                                    # exp(finite) is finite.
                                                                    # Final nan_to_num should not change anything here.
    # Relaxing tolerance slightly for floating point variations
    assert_allclose(sefa_score, expected_sefa_nan_handling, rtol=1e-4, atol=1e-5) 