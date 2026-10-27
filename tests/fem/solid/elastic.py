from conspire.constitutive.solid.elastic import AlmansiHamelEulerian
from helpers import (
    assert_nodal_forces_zero,
    assert_nodal_stiffnesses_finite_difference,
    build,
    make_block,
)
from pytest import raises
import pytest

MODELS = [
    ("AlmansiHamelEulerian", AlmansiHamelEulerian, {}),
]


@pytest.fixture(params=MODELS, ids=[name for name, _, _ in MODELS])
def model_case(request):
    return request.param


@pytest.fixture
def model(model_case, bulk_modulus, shear_modulus):
    _, cls, extra = model_case
    return build(cls, bulk_modulus, shear_modulus, extra)


@pytest.fixture
def block(model, connectivity, reference_coordinates):
    return make_block(model, connectivity, reference_coordinates)


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
