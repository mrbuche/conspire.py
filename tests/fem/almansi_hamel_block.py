from conspire.constitutive.solid.elastic import AlmansiHamel
from conspire.fem import Block
from helpers import assert_nodal_forces_zero, assert_nodal_stiffnesses_finite_difference
from pytest import raises
import pytest


@pytest.fixture
def model(bulk_modulus, shear_modulus):
    return AlmansiHamel(bulk_modulus, shear_modulus)


@pytest.fixture
def block(model, connectivity, reference_coordinates):
    return Block(model, connectivity, reference_coordinates)


def test_helmholtz_free_energy_undefined(block, reference_coordinates):
    with raises(
        TypeError,
        match="The Helmholtz free energy density"
        + " is undefined for elastic constitutive models.",
    ):
        block.helmholtz_free_energy(reference_coordinates)


def test_nodal_forces_zero(block, reference_coordinates):
    assert_nodal_forces_zero(block, reference_coordinates)


def test_nodal_stiffnesses_finite_difference(
    block, reference_coordinates, deformed_coordinates, epsilon
):
    assert_nodal_stiffnesses_finite_difference(
        block, reference_coordinates, deformed_coordinates, epsilon
    )
