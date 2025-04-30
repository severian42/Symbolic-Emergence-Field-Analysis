"""Tests for entropy calculation functions."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises
from scipy.stats import entropy as scipy_entropy

# Conditional import for astropy
try:
    from astropy.stats import knuth_bin_width
    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False
    knuth_bin_width = None

from sefa.entropy import (
    calculate_sliding_window_entropy,
    calculate_entropy_alignment,
    _calculate_local_entropy
)
from sefa.config import SEFAConfig

# --- Fixtures ---
@pytest.fixture
def base_config():
    """Base config with required entropy_window_size."""
    return SEFAConfig(entropy_window_size=5)

@pytest.fixture
def config_knuth(base_config):
    if not _HAS_ASTROPY:
        pytest.skip("astropy not installed, skipping Knuth tests")
    base_config.entropy_binning = 'knuth'
    return base_config

@pytest.fixture
def config_fixed(base_config):
    base_config.entropy_binning = 'fixed'
    base_config.entropy_bin_count = 3 # Use small number for easy testing
    return base_config

# --- _calculate_local_entropy Tests ---

def test_local_entropy_constant(config_fixed):
    """Test entropy of a constant signal (should be 0)."""
    window = np.ones(10) * 5.0
    local_entropy = _calculate_local_entropy(window, config_fixed)
    assert_allclose(local_entropy, 0.0)

def test_local_entropy_uniform_dist(config_fixed):
    """Test entropy of a uniform distribution (should be max log(B))."""
    # 3 bins: [1, 2), [2, 3), [3, 4]
    config_fixed.entropy_bin_count = 3
    window = np.array([1.1, 1.9, 2.1, 2.9, 3.1, 3.9]) # 2 samples per bin
    # Counts = [2, 2, 2], p = [1/3, 1/3, 1/3]
    # Entropy = -3 * (1/3 * log(1/3)) = log(3)
    expected_entropy = np.log(3)
    local_entropy = _calculate_local_entropy(window, config_fixed)
    assert_allclose(local_entropy, expected_entropy)

def test_local_entropy_knuth_constant(config_knuth):
    """Test Knuth entropy fallback for constant signal."""
    window = np.ones(10) * 5.0
    # Knuth might fail or return <=1 bin, should fallback and give ~0
    local_entropy = _calculate_local_entropy(window, config_knuth)
    assert_allclose(local_entropy, 0.0, atol=1e-9)

def test_local_entropy_knuth_simple(config_knuth):
    """Test Knuth entropy for a simple bimodal distribution."""
    # Expect Knuth to potentially identify 2 bins
    window = np.concatenate([np.random.normal(0, 0.1, 20), np.random.normal(1, 0.1, 20)])
    local_entropy = _calculate_local_entropy(window, config_knuth)
    # Entropy should be close to log(2) if bins are well-separated
    assert_allclose(local_entropy, np.log(2), rtol=0.2) # Relaxed tolerance

def test_local_entropy_nan(config_fixed):
    """Test that NaNs are handled in local entropy."""
    window = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    local_entropy = _calculate_local_entropy(window, config_fixed)
    # Should compute entropy based on [1, 2, 3, 4]
    assert np.isfinite(local_entropy)

def test_local_entropy_too_small(config_fixed):
    """Test local entropy returns NaN for < 2 valid points."""
    window1 = np.array([1.0, np.nan])
    window0 = np.array([np.nan, np.nan])
    assert np.isnan(_calculate_local_entropy(window1, config_fixed))
    assert np.isnan(_calculate_local_entropy(window0, config_fixed))

# --- calculate_sliding_window_entropy Tests ---

def test_sliding_entropy_basic(config_fixed):
    """Test basic sliding window entropy calculation."""
    data = np.array([1, 1, 1, 5, 5, 5, 1, 1, 1]) # Len 9
    config = config_fixed
    config.entropy_window_size = 3
    # Window centers: indices 1 to 7
    # Win 1 (0-2): [1,1,1] -> H=0
    # Win 2 (1-3): [1,1,5] -> H>0
    # Win 3 (2-4): [1,5,5] -> H>0
    # Win 4 (3-5): [5,5,5] -> H=0
    # Win 5 (4-6): [5,5,1] -> H>0
    # Win 6 (5-7): [5,1,1] -> H>0
    # Win 7 (6-8): [1,1,1] -> H=0
    entropy_S = calculate_sliding_window_entropy(data, config)

    assert len(entropy_S) == len(data)
    # Check boundary NaNs (window size 3 -> 1 NaN at each end)
    assert np.isnan(entropy_S[0])
    assert np.isnan(entropy_S[-1])
    # Check calculated values
    assert_allclose(entropy_S[1], 0.0)
    assert entropy_S[2] > 0
    assert entropy_S[3] > 0
    assert_allclose(entropy_S[4], 0.0)
    assert entropy_S[5] > 0
    assert entropy_S[6] > 0
    assert_allclose(entropy_S[7], 0.0)

def test_sliding_entropy_invalid_window(config_fixed):
    """Test error handling for invalid window sizes."""
    data = np.arange(10)
    config = config_fixed
    config.entropy_window_size = 11 # Too large
    with pytest.raises(ValueError, match="cannot be larger than data length"):
        calculate_sliding_window_entropy(data, config)
    config.entropy_window_size = 1 # Too small
    with pytest.raises(ValueError, match="must be at least 2"):
        calculate_sliding_window_entropy(data, config)

# --- calculate_entropy_alignment Tests ---

def test_entropy_alignment_basic():
    """Test basic entropy alignment E = 1 - S/SMax."""
    entropy_S = np.array([np.nan, 0.0, 0.5, 1.0, 0.8, 0.0, np.nan])
    # SMax = 1.0
    expected_E = np.array([np.nan, 1.0, 0.5, 0.0, 0.2, 1.0, np.nan])
    alignment_E = calculate_entropy_alignment(entropy_S)
    assert_allclose(alignment_E, expected_E, equal_nan=True)

def test_entropy_alignment_all_zero():
    """Test alignment when all entropy values are zero (SMax=0)."""
    entropy_S = np.array([np.nan, 0.0, 0.0, 0.0])
    # SMax = 0, E = 1.0 where S=0
    expected_E = np.array([np.nan, 1.0, 1.0, 1.0])
    alignment_E = calculate_entropy_alignment(entropy_S)
    assert_allclose(alignment_E, expected_E, equal_nan=True)

def test_entropy_alignment_all_nan():
    """Test alignment when all entropy values are NaN."""
    entropy_S = np.array([np.nan, np.nan, np.nan])
    expected_E = np.array([np.nan, np.nan, np.nan])
    alignment_E = calculate_entropy_alignment(entropy_S)
    assert_allclose(alignment_E, expected_E, equal_nan=True)

# TODO: Add tests for sliding window entropy and alignment score
# TODO: Test different binning methods (Knuth, fixed)
# TODO: Test small W behavior and alternative estimators (k-NN) (Limitation 2.2) 