from helpers import (
    HYPERELASTIC_MODELS,
    build,
    make_model,
    assert_nodal_forces_finite_difference,
    assert_nodal_forces_zero,
    assert_nodal_stiffnesses_finite_difference,
)
import numpy as np
import pytest

MODELS = HYPERELASTIC_MODELS


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


def test_helmholtz_free_energy_zero(fem_model, reference_coordinates):
    assert fem_model.helmholtz_free_energy(reference_coordinates) == 0


def test_helmholtz_free_energy_affine(
    fem_model, model, affinely_deformed_coordinates, deformation_gradient, abs_tol
):
    assert (
        np.abs(
            fem_model.helmholtz_free_energy(affinely_deformed_coordinates)
            - model.helmholtz_free_energy_density(deformation_gradient)
        )
        < abs_tol
    )


def test_nodal_forces_zero(fem_model, reference_coordinates):
    assert_nodal_forces_zero(fem_model, reference_coordinates)


def test_nodal_forces_finite_difference(
    fem_model, reference_coordinates, deformed_coordinates, epsilon
):
    assert_nodal_forces_finite_difference(
        fem_model, reference_coordinates, deformed_coordinates, epsilon
    )


def test_nodal_stiffnesses_finite_difference(
    fem_model, reference_coordinates, deformed_coordinates, epsilon
):
    assert_nodal_stiffnesses_finite_difference(
        fem_model, reference_coordinates, deformed_coordinates, epsilon
    )
