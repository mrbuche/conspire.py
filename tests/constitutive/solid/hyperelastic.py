from constitutive_helpers import (
    HYPERELASTIC_MODELS,
    build,
    assert_bulk_modulus,
    assert_cauchy_stress_relate_first_piola_kirchhoff_stress,
    assert_cauchy_stress_relate_second_piola_kirchhoff_stress,
    assert_cauchy_stress_symmetry,
    assert_cauchy_stress_zero,
    assert_cauchy_tangent_stiffness_finite_difference,
    assert_first_piola_kirchhoff_stress_finite_difference,
    assert_first_piola_kirchhoff_stress_zero,
    assert_first_piola_kirchhoff_tangent_stiffness_finite_difference,
    assert_first_piola_kirchhoff_tangent_stiffness_symmetry,
    assert_helmholtz_free_energy_density_zero,
    assert_second_piola_kirchhoff_stress_zero,
    assert_second_piola_kirchhoff_tangent_stiffness_finite_difference,
    assert_shear_modulus,
    assert_str,
)
import pytest

MODELS = HYPERELASTIC_MODELS


@pytest.fixture(params=MODELS, ids=[name for name, *_ in MODELS])
def model_case(request):
    return request.param


@pytest.fixture
def model(model_case):
    _, cls, params, _ = model_case
    return build(cls, params)


def test_str(model, model_case):
    name, _, params, _ = model_case
    assert_str(model, name, params)


def test_helmholtz_free_energy_density_zero(model, identity):
    assert_helmholtz_free_energy_density_zero(model, identity)


def test_first_piola_kirchhoff_stress_finite_difference(
    model, deformation_gradient, epsilon
):
    assert_first_piola_kirchhoff_stress_finite_difference(
        model, deformation_gradient, epsilon
    )


def test_first_piola_kirchhoff_tangent_stiffness_symmetry(
    model, deformation_gradient, abs_tol
):
    assert_first_piola_kirchhoff_tangent_stiffness_symmetry(
        model, deformation_gradient, abs_tol
    )


def test_cauchy_stress_zero(model, identity, zero):
    assert_cauchy_stress_zero(model, identity, zero)


def test_first_piola_kirchhoff_stress_zero(model, identity, zero):
    assert_first_piola_kirchhoff_stress_zero(model, identity, zero)


def test_second_piola_kirchhoff_stress_zero(model, identity, zero):
    assert_second_piola_kirchhoff_stress_zero(model, identity, zero)


def test_cauchy_stress_symmetry(model, deformation_gradient, abs_tol):
    assert_cauchy_stress_symmetry(model, deformation_gradient, abs_tol)


def test_cauchy_stress_relate_first_piola_kirchhoff_stress(
    model, deformation_gradient, abs_tol
):
    assert_cauchy_stress_relate_first_piola_kirchhoff_stress(
        model, deformation_gradient, abs_tol
    )


def test_cauchy_stress_relate_second_piola_kirchhoff_stress(
    model, deformation_gradient, abs_tol
):
    assert_cauchy_stress_relate_second_piola_kirchhoff_stress(
        model, deformation_gradient, abs_tol
    )


def test_shear_modulus(model, model_case, simple_shear_small, epsilon):
    shear_modulus = model_case[3]
    if shear_modulus is None:
        pytest.skip("no closed-form shear modulus for this parameter set")
    assert_shear_modulus(model, simple_shear_small, epsilon, shear_modulus)


def test_bulk_modulus(model, volumetric_small, epsilon, bulk_modulus):
    assert_bulk_modulus(model, volumetric_small, epsilon, bulk_modulus)


def test_cauchy_tangent_stiffness_finite_difference(
    model, deformation_gradient, abs_tol, epsilon
):
    assert_cauchy_tangent_stiffness_finite_difference(
        model, deformation_gradient, abs_tol, epsilon
    )


def test_first_piola_kirchhoff_tangent_stiffness_finite_difference(
    model, deformation_gradient, epsilon
):
    assert_first_piola_kirchhoff_tangent_stiffness_finite_difference(
        model, deformation_gradient, epsilon
    )


def test_second_piola_kirchhoff_tangent_stiffness_finite_difference(
    model, deformation_gradient, epsilon
):
    assert_second_piola_kirchhoff_tangent_stiffness_finite_difference(
        model, deformation_gradient, epsilon
    )
