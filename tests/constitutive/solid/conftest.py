from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

_ABS_TOL = 1e-12
_EPSILON = 1e-6
_BULK_MODULUS = 13
_SHEAR_MODULUS = 3
_DEFORMATION_GRADIENT = np.array(
    [
        [0.63595746, 0.69157849, 0.71520784],
        [0.80589604, 0.83687323, 0.19312595],
        [0.05387420, 0.86551549, 0.41880244],
    ]
)


@pytest.fixture
def abs_tol():
    return _ABS_TOL


@pytest.fixture
def epsilon():
    return _EPSILON


@pytest.fixture
def bulk_modulus():
    return _BULK_MODULUS


@pytest.fixture
def shear_modulus():
    return _SHEAR_MODULUS


@pytest.fixture
def zero():
    return np.zeros((3, 3))


@pytest.fixture
def identity():
    return np.eye(3)


@pytest.fixture
def deformation_gradient():
    return _DEFORMATION_GRADIENT.copy()


@pytest.fixture
def simple_shear_small(epsilon):
    return np.array([[1, epsilon, 0], [0, 1, 0], [0, 0, 1]])


@pytest.fixture
def volumetric_small(identity, epsilon):
    return identity * (1 + epsilon) ** (1 / 3)
