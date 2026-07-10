from conspire.math.special import (
    inverse_langevin,
    lambert_w,
    langevin,
    langevin_derivative,
    rosenbrock,
    sinhc,
)
import numpy as np

ABS_TOL = 1e-10
EPSILON = 1e-6


def test_lambert_w_zero():
    assert lambert_w(0.0) == 0.0


def test_lambert_w_end():
    assert lambert_w(-1.0 / np.e) == -1.0


def test_lambert_w_euler():
    assert lambert_w(np.e) == 1.0


def test_lambert_w_identity():
    for x in np.linspace(-1.0 / np.e, 10.0, 1000):
        w = lambert_w(x)
        assert np.abs(w * np.exp(w) - x) < ABS_TOL


def test_langevin_zero():
    assert langevin(0.0) == 0.0


def test_langevin_derivative_zero():
    assert langevin_derivative(0.0) == 1.0 / 3.0


def test_langevin_derivative_finite_difference():
    for x in np.linspace(-5.0, 5.0, 1000):
        if np.abs(x) < 0.05:
            continue
        d_langevin = (langevin(x + EPSILON / 2) - langevin(x - EPSILON / 2)) / EPSILON
        assert np.abs(langevin_derivative(x) - d_langevin) < EPSILON


def test_inverse_langevin_zero():
    assert inverse_langevin(0.0) == 0.0


def test_inverse_langevin_identity():
    for y in np.linspace(-0.999, 0.999, 1000):
        assert np.abs(langevin(inverse_langevin(y)) - y) < ABS_TOL


def test_inverse_langevin_identity_small():
    for y in np.linspace(-3e-3, 3e-3, 100):
        assert np.abs(langevin(inverse_langevin(y)) - y) < ABS_TOL


def test_sinhc_zero():
    assert sinhc(0.0) == 1.0


def test_sinhc():
    for x in np.linspace(-5.0, 5.0, 1000):
        if x == 0.0:
            continue
        assert np.abs(sinhc(x) - np.sinh(x) / x) < ABS_TOL


def test_rosenbrock_zero():
    for b in [1.0, 10.0, 100.0]:
        for n in [2, 3, 5]:
            assert rosenbrock([1.0] * n, 1.0, b) == 0.0


def test_rosenbrock():
    x = [0.5, 0.25, -0.75, 1.5]
    a = 1.0
    b = 100.0
    expected = sum(
        (a - x[i]) ** 2 + b * (x[i + 1] - x[i] ** 2) ** 2 for i in range(len(x) - 1)
    )
    assert np.abs(rosenbrock(x, a, b) - expected) < ABS_TOL
