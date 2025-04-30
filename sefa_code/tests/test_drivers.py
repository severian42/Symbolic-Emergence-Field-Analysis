"""Tests for the driver weighting and field construction module."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from sefa.drivers import calculate_driver_weights

# TODO: Add tests for weight calculation and field construction 

def test_calculate_weights_beta_2():
    """Test weight calculation with default beta=2."""
    gamma_k = np.array([0.0, 1.0, 2.0, -3.0])
    expected_weights = np.array([
        1.0 / (1 + 0**2),
        1.0 / (1 + 1**2),
        1.0 / (1 + 2**2),
        1.0 / (1 + (-3)**2)
    ]) # [1.0, 0.5, 0.2, 0.1]

    weights = calculate_driver_weights(gamma_k, beta=2.0)
    assert_allclose(weights, expected_weights)

def test_calculate_weights_beta_1():
    """Test weight calculation with beta=1."""
    gamma_k = np.array([0.0, 1.0, 4.0, -9.0])
    # Formula uses abs(gamma_k) before exponentiation
    expected_weights = np.array([
        1.0 / (1 + np.abs(0.0)**1),
        1.0 / (1 + np.abs(1.0)**1),
        1.0 / (1 + np.abs(4.0)**1),
        1.0 / (1 + np.abs(-9.0)**1)
    ]) # [1.0, 0.5, 0.2, 0.1]

    weights = calculate_driver_weights(gamma_k, beta=1.0)
    assert_allclose(weights, expected_weights)

def test_calculate_weights_beta_float():
    """Test weight calculation with float beta."""
    gamma_k = np.array([0.0, 2.0])
    beta = 1.5
    expected_weights = np.array([
        1.0 / (1 + 0**beta),
        1.0 / (1 + 2**beta)
    ])
    weights = calculate_driver_weights(gamma_k, beta=beta)
    assert_allclose(weights, expected_weights)

def test_calculate_weights_empty():
    """Test weight calculation with empty input."""
    gamma_k = np.array([])
    expected_weights = np.array([])
    weights = calculate_driver_weights(gamma_k, beta=2.0)
    assert_allclose(weights, expected_weights)

def test_calculate_weights_invalid_beta():
    """Test that calculation fails with non-positive beta."""
    gamma_k = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="beta must be positive"):
        calculate_driver_weights(gamma_k, beta=0.0)
    with pytest.raises(ValueError, match="beta must be positive"):
        calculate_driver_weights(gamma_k, beta=-1.0) 