"""Tests for SEFA score thresholding functions."""

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose

# Import the function to test
from sefa.thresholding import apply_threshold

# Mocking optional dependencies
class MockMissingModule:
    pass

try:
    import skimage
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    skimage = MockMissingModule()
    skimage.filters = MockMissingModule()
    skimage.filters.threshold_otsu = None

try:
    import sklearn
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    sklearn = MockMissingModule()
    sklearn.mixture = MockMissingModule()
    sklearn.mixture.GaussianMixture = None

# --- Fixtures ---
@pytest.fixture
def sample_score():
    """A sample SEFA score array with some separation."""
    # Bimodal-like data: background around 0.1, signal around 0.8
    np.random.seed(0)
    background = np.random.normal(0.1, 0.05, 80)
    signal = np.random.normal(0.8, 0.1, 20)
    score = np.concatenate([background, signal])
    np.random.shuffle(score)
    score = np.clip(score, 0, 1) # Ensure non-negative
    # Add a NaN
    score[50] = np.nan
    return score

# --- Test Functions ---

def test_apply_threshold_manual(sample_score):
    """Test manual thresholding."""
    threshold = 0.5
    mask = apply_threshold(sample_score, method='manual', threshold_value=threshold)
    expected_mask = np.where(~np.isnan(sample_score), sample_score > threshold, False)
    assert_equal(mask, expected_mask)

def test_apply_threshold_percentile(sample_score):
    """Test percentile thresholding."""
    percentile = 80 # Expect threshold near start of signal distribution
    mask = apply_threshold(sample_score, method='percentile', percentile=percentile)
    valid_score = sample_score[~np.isnan(sample_score)]
    expected_threshold = np.percentile(valid_score, percentile)
    expected_mask = np.where(~np.isnan(sample_score), sample_score > expected_threshold, False)
    assert_equal(mask, expected_mask)
    # Check roughly correct number are above threshold
    assert np.sum(mask) > 15 and np.sum(mask) < 25 # Around 20% should be above

@pytest.mark.skipif(not _HAS_SKIMAGE, reason="scikit-image not installed")
def test_apply_threshold_otsu(sample_score):
    """Test Otsu thresholding (requires scikit-image)."""
    mask = apply_threshold(sample_score, method='otsu')
    assert mask is not None
    assert mask.dtype == bool
    assert len(mask) == len(sample_score)
    # Otsu should find a threshold between the two modes (~0.4-0.5)
    otsu_threshold = skimage.filters.threshold_otsu(sample_score[~np.isnan(sample_score)])
    expected_mask = np.where(~np.isnan(sample_score), sample_score > otsu_threshold, False)
    assert_equal(mask, expected_mask)
    assert np.sum(mask) > 15 and np.sum(mask) < 25 # Should select the signal points

@pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
def test_apply_threshold_mog(sample_score):
    """Test Gaussian Mixture Model thresholding (requires scikit-learn)."""
    # Note: Threshold logic is heuristic in current implementation
    mask = apply_threshold(sample_score, method='mog', n_components=2)
    assert mask is not None
    assert mask.dtype == bool
    assert len(mask) == len(sample_score)
    # Cannot easily verify the exact heuristic threshold, but check basics
    assert np.sum(mask) > 0 and np.sum(mask) < len(sample_score) # Should find some separation

def test_apply_threshold_missing_deps(sample_score, monkeypatch):
    """Test graceful failure when optional dependencies are missing."""
    # Test Otsu failure
    monkeypatch.setattr("sefa.thresholding._HAS_SKIMAGE", False)
    monkeypatch.setattr("sefa.thresholding.threshold_otsu", None)
    mask_otsu = apply_threshold(sample_score, method='otsu')
    assert mask_otsu is None

    # Test MoG failure
    monkeypatch.setattr("sefa.thresholding._HAS_SKLEARN", False)
    monkeypatch.setattr("sefa.thresholding.GaussianMixture", None)
    mask_mog = apply_threshold(sample_score, method='mog')
    assert mask_mog is None

def test_apply_threshold_all_nan():
    """Test thresholding when input is all NaNs."""
    score = np.full(10, np.nan)
    mask = apply_threshold(score, method='manual', threshold_value=0.5)
    expected_mask = np.zeros_like(score, dtype=bool)
    assert_equal(mask, expected_mask)

@pytest.mark.skipif(not _HAS_SKIMAGE, reason="scikit-image not installed")
def test_apply_threshold_constant_otsu(monkeypatch):
    """Test Otsu behavior with constant data."""
    # monkeypatch.setattr("sefa.thresholding._HAS_SKIMAGE", True) # No longer needed with skipif
    # monkeypatch.setattr("sefa.thresholding.threshold_otsu", skimage.filters.threshold_otsu)
    score_pos = np.ones(10) * 0.8
    score_zero = np.zeros(10)

    with pytest.warns(UserWarning, match="Otsu threshold is undefined"):
        mask_pos = apply_threshold(score_pos, method='otsu')
    assert_equal(mask_pos, np.ones_like(score_pos, dtype=bool))

    with pytest.warns(UserWarning, match="Otsu threshold is undefined"):
        mask_zero = apply_threshold(score_zero, method='otsu')
    assert_equal(mask_zero, np.zeros_like(score_zero, dtype=bool))

def test_apply_threshold_invalid_method(sample_score):
    """Test failure with unknown method."""
    mask = apply_threshold(sample_score, method='unknown')
    assert mask is None

def test_apply_threshold_missing_kwargs(sample_score):
    """Test failure when required kwargs are missing."""
    mask_p = apply_threshold(sample_score, method='percentile') # Missing percentile kwarg
    assert mask_p is None
    mask_m = apply_threshold(sample_score, method='manual') # Missing threshold_value kwarg
    assert mask_m is None

# TODO: Add tests for Otsu, percentile, GMM thresholding 