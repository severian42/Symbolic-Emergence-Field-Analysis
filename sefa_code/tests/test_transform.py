"""Tests for field construction and analytic signal generation."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.signal import hilbert

from sefa.transform import construct_field, get_analytic_signal
from sefa.config import SEFAConfig

# --- Fixtures ---
@pytest.fixture
def default_config():
    return SEFAConfig(entropy_window_size=51) # Window size needed for validation

@pytest.fixture
def simple_signal_setup():
    """Setup for a simple cosine signal."""
    ymin, ymax, num_points = 0, 2*np.pi, 100
    domain_y = np.linspace(ymin, ymax, num_points, endpoint=False)
    gamma_k = np.array([1.0]) # Single driver, frequency = 1
    weights_w = np.array([1.0]) # Weight = 1
    field_V0 = np.cos(domain_y) # Expected field
    return domain_y, gamma_k, weights_w, field_V0

# --- construct_field Tests ---

def test_construct_field_single_driver(simple_signal_setup):
    """Test field construction with a single cosine driver."""
    domain_y, gamma_k, weights_w, expected_field = simple_signal_setup
    constructed_field = construct_field(domain_y, gamma_k, weights_w)
    assert_allclose(constructed_field, expected_field, atol=1e-9)

def test_construct_field_multiple_drivers():
    """Test field construction with multiple drivers."""
    domain_y = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    gamma_k = np.array([1.0, 2.0])
    weights_w = np.array([0.5, 0.2])
    # V0 = 0.5*cos(1*y) + 0.2*cos(2*y)
    expected_field = np.array([
        0.5*1 + 0.2*1,    # y=0
        0.5*0 + 0.2*(-1), # y=pi/2
        0.5*(-1) + 0.2*1,# y=pi
        0.5*0 + 0.2*(-1) # y=3pi/2
    ]) # [0.7, -0.2, -0.3, -0.2]
    constructed_field = construct_field(domain_y, gamma_k, weights_w)
    assert_allclose(constructed_field, expected_field)

def test_construct_field_mismatch_drivers_weights():
    """Test error handling when drivers and weights lengths differ."""
    domain_y = np.linspace(0, 1, 10)
    gamma_k = np.array([1.0, 2.0])
    weights_w = np.array([0.5])
    with pytest.raises(ValueError, match="must match number of weights"):
        construct_field(domain_y, gamma_k, weights_w)

# --- get_analytic_signal Tests ---

def test_get_analytic_signal_periodic(default_config, simple_signal_setup):
    """Test analytic signal with periodic boundary handling."""
    domain_y, gamma_k, weights_w, field_V0 = simple_signal_setup
    config = default_config
    config.boundary_method = 'periodic'

    analytic_signal, indices = get_analytic_signal(field_V0, config)

    # For V0 = cos(y), analytic signal should be cos(y) + i*sin(y) = exp(i*y)
    # Hilbert transform of cos(y) is sin(y)
    expected_analytic = np.exp(1j * domain_y)

    # Check shape and type
    assert_equal(analytic_signal.shape, field_V0.shape)
    assert np.iscomplexobj(analytic_signal)
    # Compare with theoretical result (FFT-based Hilbert might have small errors)
    assert_allclose(analytic_signal, expected_analytic, atol=1e-9)
    # Indices should match original length for periodic
    assert_equal(indices, np.arange(len(field_V0)))


def test_get_analytic_signal_discard(default_config, simple_signal_setup):
    """Test analytic signal with discard boundary handling (padding happens internally)."""
    domain_y, gamma_k, weights_w, field_V0 = simple_signal_setup
    config = default_config
    config.boundary_method = 'discard'
    config.boundary_discard_fraction = 0.1 # Discard 10%

    # get_analytic_signal returns the full signal before discarding
    analytic_signal_full, indices = get_analytic_signal(field_V0, config)

    # The actual discarding happens in handle_boundaries, called by SEFA class
    # Here, we just check the output of get_analytic_signal itself
    expected_analytic = hilbert(field_V0) # Standard periodic Hilbert for comparison

    assert_equal(analytic_signal_full.shape, field_V0.shape)
    assert np.iscomplexobj(analytic_signal_full)
    # Check that indices match original length, as discarding isn't done here
    assert_equal(indices, np.arange(len(field_V0)))
    # The result should be close to the standard Hilbert transform
    assert_allclose(analytic_signal_full, expected_analytic, atol=1e-9)


def test_get_analytic_signal_mirror(default_config, simple_signal_setup):
    """Test analytic signal with mirror boundary handling."""
    domain_y, gamma_k, weights_w, field_V0 = simple_signal_setup
    config = default_config
    config.boundary_method = 'mirror'
    config.boundary_discard_fraction = 0.1 # Determines pad width
    pad_width = int(len(field_V0) * config.boundary_discard_fraction) # 10

    analytic_signal_padded, original_indices_in_padded = get_analytic_signal(field_V0, config)

    # Check shape of padded signal
    assert_equal(analytic_signal_padded.shape, (len(field_V0) + 2 * pad_width,))
    assert np.iscomplexobj(analytic_signal_padded)

    # Check indices cover the original range within the padded signal
    assert_equal(len(original_indices_in_padded), len(field_V0))
    assert_equal(original_indices_in_padded[0], pad_width)
    assert_equal(original_indices_in_padded[-1], pad_width + len(field_V0) - 1)

    # Verify the content by extracting the original part
    analytic_signal_original_part = analytic_signal_padded[original_indices_in_padded]
    # Compare this to theory or periodic hilbert? Mirror padding changes the result.
    # For cos(x) on [0, 2pi), mirror padding creates discontinuities.
    # Just check if the calculation ran without error and shapes are correct.


def test_get_analytic_signal_mirror_zero_pad(default_config, simple_signal_setup):
    """Test mirror padding when fraction results in zero pad width."""
    domain_y, gamma_k, weights_w, field_V0 = simple_signal_setup
    config = default_config
    config.boundary_method = 'mirror'
    config.boundary_discard_fraction = 0.0 # Zero pad width

    # Should warn
    with pytest.warns(UserWarning, match="results in zero padding width"):
        analytic_signal, indices = get_analytic_signal(field_V0, config)

    # Result should be same as periodic
    expected_analytic = hilbert(field_V0)
    assert_allclose(analytic_signal, expected_analytic, atol=1e-9)
    assert_equal(indices, np.arange(len(field_V0)))

# TODO: Add tests for Hilbert transform properties and analytic signal
# TODO: Test padding/windowing effects (Limitation 2.1) 