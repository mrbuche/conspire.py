from conspire.constitutive.solid.hyperelastic import Fung
from conspire.fem import Block
from helpers import (
    assert_nodal_forces_finite_difference,
    assert_nodal_forces_zero,
    assert_nodal_stiffnesses_finite_difference,
)
import numpy as np
import pytest

EXTRA_MODULUS = 1
EXPONENT = 1


@pytest.fixture
def model(bulk_modulus, shear_modulus):
    return Fung(bulk_modulus, shear_modulus, EXTRA_MODULUS, EXPONENT)


@pytest.fixture
def block(model, connectivity, reference_coordinates):
    return Block(model, connectivity, reference_coordinates)


def test_helmholtz_free_energy_zero(block, reference_coordinates):
    assert block.helmholtz_free_energy(reference_coordinates) == 0


def test_helmholtz_free_energy_affine(
    block, model, affinely_deformed_coordinates, deformation_gradient, abs_tol
):
    assert (
        np.abs(
            block.helmholtz_free_energy(affinely_deformed_coordinates)
            - model.helmholtz_free_energy_density(deformation_gradient)
        )
        < abs_tol
    )


def test_nodal_forces_zero(block, reference_coordinates):
    assert_nodal_forces_zero(block, reference_coordinates)


def test_nodal_forces_finite_difference(
    block, reference_coordinates, deformed_coordinates, epsilon
):
    assert_nodal_forces_finite_difference(
        block, reference_coordinates, deformed_coordinates, epsilon
    )


def test_nodal_stiffnesses_finite_difference(
    block, reference_coordinates, deformed_coordinates, epsilon
):
    assert_nodal_stiffnesses_finite_difference(
        block, reference_coordinates, deformed_coordinates, epsilon
    )
