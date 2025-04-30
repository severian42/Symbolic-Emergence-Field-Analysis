"""Tests for the utility functions module."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from sefa.utils import (
    mirror_pad,
    handle_boundaries,
    calculate_derivative,
    _finite_difference, # Test internal helpers too
    _savgol_derivative
    # _polyfit_derivative # Cannot test directly as it falls back
)
from sefa.config import SEFAConfig, SavgolParams, PolyfitParams

# TODO: Add tests for phase unwrapping
# TODO: Add tests for padding/windowing functions
# TODO: Add tests for derivative smoothing (Savitzky-Golay)
# TODO: Add tests for adaptive binning (Knuth)
# TODO: Add tests for k-NN entropy estimator 

# --- Fixtures ---
@pytest.fixture
def default_config():
    """Provides a default SEFAConfig instance for tests."""
    # Need to set entropy_window_size for validation
    return SEFAConfig(entropy_window_size=51)

@pytest.fixture
def sample_data():
    """Provides simple 1D data for testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

# --- mirror_pad Tests ---

def test_mirror_pad_basic(sample_data):
    """Test basic mirror padding."""
    padded, info = mirror_pad(sample_data, pad_width=2)
    # Expect: [3, 2, | 1, 2, 3, 4, 5, 6 | , 5, 4]
    expected = np.array([3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0])
    assert_allclose(padded, expected)
    assert info == {'left': 2, 'right': 2}

def test_mirror_pad_zero_width(sample_data):
    """Test mirror padding with zero width."""
    padded, info = mirror_pad(sample_data, pad_width=0)
    assert_allclose(padded, sample_data)
    assert info == {'left': 0, 'right': 0}

def test_mirror_pad_negative_width(sample_data):
    """Test mirror padding with negative width (should behave like zero)."""
    padded, info = mirror_pad(sample_data, pad_width=-1)
    assert_allclose(padded, sample_data)
    assert info == {'left': 0, 'right': 0}

# --- handle_boundaries Tests ---

def test_handle_boundaries_discard(default_config, sample_data):
    """Test boundary handling with discard method."""
    config = default_config
    config.boundary_method = 'discard'
    config.boundary_discard_fraction = 0.2 # Discard 20% -> floor(6*0.2)=1 from each side
    handled = handle_boundaries(sample_data, config, None)
    expected = np.array([2.0, 3.0, 4.0, 5.0]) # Discard 1, 6
    assert_allclose(handled, expected)

def test_handle_boundaries_discard_zero_fraction(default_config, sample_data):
    """Test discard boundary handling with zero fraction."""
    config = default_config
    config.boundary_method = 'discard'
    config.boundary_discard_fraction = 0.0
    handled = handle_boundaries(sample_data, config, None)
    assert_allclose(handled, sample_data)

def test_handle_boundaries_mirror(default_config, sample_data):
    """Test boundary handling with mirror method (extracting original part)."""
    config = default_config
    config.boundary_method = 'mirror'
    # Simulate data that was padded
    padded_data = np.array([3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0])
    original_indices = np.arange(2, 8) # Indices of original data within padded
    handled = handle_boundaries(padded_data, config, original_indices)
    assert_allclose(handled, sample_data)

def test_handle_boundaries_mirror_no_indices(default_config, sample_data):
    """Test mirror boundary handling when indices are missing (should warn and return full)."""
    config = default_config
    config.boundary_method = 'mirror'
    with pytest.warns(UserWarning, match="no original_indices provided"):
        handled = handle_boundaries(sample_data, config, None)
    assert_allclose(handled, sample_data)

def test_handle_boundaries_periodic(default_config, sample_data):
    """Test periodic boundary handling (should be a no-op)."""
    config = default_config
    config.boundary_method = 'periodic'
    handled = handle_boundaries(sample_data, config, None)
    assert_allclose(handled, sample_data)

# --- Derivative Tests ---

@pytest.fixture
def deriv_data():
    """Quadratic data for testing derivatives."""
    # y = x^2 + 1, dy = 1
    # y' = 2x
    # y'' = 2
    x = np.arange(7.0)
    y = x**2 + 1.0
    dy = 1.0
    return x, y, dy

def test_finite_difference_order1(deriv_data):
    """Test finite difference first derivative."""
    x, y, dy = deriv_data
    deriv1 = _finite_difference(y, dy, order=1)
    expected_deriv1 = 2 * x # Exact derivative
    # np.gradient will have errors at boundaries
    assert_allclose(deriv1[1:-1], expected_deriv1[1:-1], atol=1e-9)
    # Check boundaries separately (less accurate)
    assert np.isclose(deriv1[0], 1.0) # Forward diff: (y[1]-y[0])/dy = (2-1)/1=1
    assert np.isclose(deriv1[-1], 11.0) # Backward diff: (y[-1]-y[-2])/dy = (37-26)/1=11

def test_finite_difference_order2(deriv_data):
    """Test finite difference second derivative."""
    x, y, dy = deriv_data
    deriv2 = _finite_difference(y, dy, order=2)
    expected_deriv2 = np.full_like(x, 2.0)
    # np.gradient(np.gradient()) is not perfectly accurate, esp. at boundaries
    assert_allclose(deriv2[1:-1], expected_deriv2[1:-1], atol=1e-9)
    # Boundaries will be less accurate

def test_savgol_derivative_order1(deriv_data):
    """Test Savitzky-Golay first derivative."""
    x, y, dy = deriv_data
    params = SavgolParams(window_length=5, polyorder=2, deriv_order=1)
    deriv1 = _savgol_derivative(y, dy, params)
    expected_deriv1 = 2 * x
    # Savgol should be more accurate, especially if polyorder matches data
    assert_allclose(deriv1, expected_deriv1, atol=1e-9)

def test_savgol_derivative_order2(deriv_data):
    """Test Savitzky-Golay second derivative."""
    x, y, dy = deriv_data
    params = SavgolParams(window_length=5, polyorder=2, deriv_order=2)
    deriv2 = _savgol_derivative(y, dy, params)
    expected_deriv2 = np.full_like(x, 2.0)
    assert_allclose(deriv2, expected_deriv2, atol=1e-9)

def test_savgol_derivative_low_window(deriv_data):
    """Test Savitzky-Golay fallback with too small window."""
    x, y, dy = deriv_data
    params = SavgolParams(window_length=3, polyorder=3, deriv_order=1) # window < polyorder+1
    # Should warn and fall back to finite difference
    with pytest.warns(UserWarning, match="Falling back to finite difference"):
        deriv1 = _savgol_derivative(y, dy, params)
    # Check if it matches finite difference result
    expected_finite_diff = _finite_difference(y, dy, order=1)
    assert_allclose(deriv1, expected_finite_diff)

def test_calculate_derivative_dispatch(default_config, deriv_data):
    """Test that calculate_derivative dispatches correctly."""
    x, y, dy = deriv_data
    config = default_config

    # Finite Difference
    config.derivative_method = 'finite_difference'
    deriv_fd = calculate_derivative(y, dy, config, deriv_order=1)
    expected_fd = _finite_difference(y, dy, order=1)
    assert_allclose(deriv_fd, expected_fd)

    # Savitzky-Golay
    config.derivative_method = 'savgol'
    config.savgol_frequency_params = SavgolParams(window_length=5, polyorder=2, deriv_order=1)
    deriv_sg = calculate_derivative(y, dy, config, deriv_order=1)
    expected_sg = _savgol_derivative(y, dy, config.savgol_frequency_params)
    assert_allclose(deriv_sg, expected_sg)

    # Polyfit (should fallback to Savgol)
    config.derivative_method = 'polyfit'
    config.polyfit_frequency_params = PolyfitParams(window_length=5, polyorder=2, deriv_order=1)
    with pytest.warns(UserWarning, match="_polyfit_derivative is not yet implemented"):
         deriv_pf = calculate_derivative(y, dy, config, deriv_order=1)
    # Expected is Savgol result with corresponding params
    savgol_params_equiv = SavgolParams(window_length=5, polyorder=2, deriv_order=1)
    expected_pf_fallback = _savgol_derivative(y, dy, savgol_params_equiv)
    assert_allclose(deriv_pf, expected_pf_fallback)

def test_calculate_derivative_invalid_method(default_config, deriv_data):
    """Test calculate_derivative fails with unknown method."""
    x, y, dy = deriv_data
    config = default_config
    config.derivative_method = 'unknown'
    with pytest.raises(ValueError, match="Unknown derivative_method"):
        calculate_derivative(y, dy, config, deriv_order=1) 