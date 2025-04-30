"""Tests for the domain discretization functions."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from sefa.domain import discretize_domain

def test_discretize_domain_basic():
    """Test basic domain discretization."""
    ymin, ymax, num_points = 0.0, 10.0, 11
    domain_y, dy = discretize_domain(ymin, ymax, num_points)

    expected_domain = np.linspace(ymin, ymax, num_points)
    expected_dy = 1.0

    assert_equal(domain_y.shape, (num_points,))
    assert_allclose(domain_y, expected_domain)
    assert_allclose(dy, expected_dy)

def test_discretize_domain_float():
    """Test domain discretization with float endpoints."""
    ymin, ymax, num_points = np.pi, 2*np.pi, 5
    domain_y, dy = discretize_domain(ymin, ymax, num_points)

    expected_domain = np.linspace(ymin, ymax, num_points)
    expected_dy = (2*np.pi - np.pi) / (num_points - 1)

    assert_equal(domain_y.shape, (num_points,))
    assert_allclose(domain_y, expected_domain)
    assert_allclose(dy, expected_dy)

def test_discretize_domain_min_points():
    """Test domain discretization with minimum number of points (2)."""
    ymin, ymax, num_points = 0.0, 1.0, 2
    domain_y, dy = discretize_domain(ymin, ymax, num_points)

    expected_domain = np.array([0.0, 1.0])
    expected_dy = 1.0

    assert_equal(domain_y.shape, (num_points,))
    assert_allclose(domain_y, expected_domain)
    assert_allclose(dy, expected_dy)


def test_discretize_domain_insufficient_points():
    """Test that discretization fails with fewer than 2 points."""
    with pytest.raises(ValueError, match="must be at least 2"):
        discretize_domain(0.0, 1.0, 1)
    with pytest.raises(ValueError, match="must be at least 2"):
        discretize_domain(0.0, 1.0, 0)
    with pytest.raises(ValueError, match="must be at least 2"):
        discretize_domain(0.0, 1.0, -5)

# TODO: Add tests for normal cases and edge cases (e.g., M=2) 