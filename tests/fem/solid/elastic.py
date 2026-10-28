from helpers import (
    ELASTIC_MODELS,
    assert_nodal_forces_zero,
    assert_nodal_stiffnesses_finite_difference,
    build,
    make_model,
)
from pytest import raises
import pytest

MODELS = ELASTIC_MODELS


@pytest.fixture(params=MODELS, ids=[name for name, *_ in MODELS])
def model_case(request):
    return request.param


@pytest.fixture
def model(model_case):
    _, cls, params, _ = model_case
    return build(cls, params)


@pytest.fixture
def fem_model(model, connectivity, reference_coordinates):
    return make_model(model, connectivity, reference_coordinates)


def test_helmholtz_free_energy_undefined(fem_model, reference_coordinates):
    with raises(
        TypeError,
        match="The Helmholtz free energy density"
        + " is undefined for elastic constitutive models.",
    ):
        fem_model.helmholtz_free_energy(reference_coordinates)


def test_nodal_forces_zero(fem_model, reference_coordinates):
    assert_nodal_forces_zero(fem_model, reference_coordinates)


def test_nodal_stiffnesses_finite_difference(
    fem_model, reference_coordinates, deformed_coordinates, epsilon
):
    assert_nodal_stiffnesses_finite_difference(
        fem_model, reference_coordinates, deformed_coordinates, epsilon
    )
