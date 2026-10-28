from conspire.constitutive.solid import elastic, hyperelastic
from conspire.units import Stress
import numpy as np

_BS = {"bulk_modulus": 13, "shear_modulus": 3}


def _models(module, cases):
    return [
        (name, getattr(module, name), params, shear) for name, params, shear in cases
    ]


ELASTIC_MODELS = _models(
    elastic,
    [
        (name, {**_BS, **extra}, 3)
        for name, extra in [
            ("AlmansiHamelEulerian", {}),
            ("AlmansiHamelLagrangian", {}),
            ("BazantItskovEulerian", {"exponent": 1.3}),
            ("BazantItskovLagrangian", {"exponent": 1.3}),
            ("Hencky", {}),
            ("SaintVenantKirchhoff", {}),
            ("SethHillEulerian", {"exponent": 1.3}),
            ("SethHillLagrangian", {"exponent": 1.3}),
        ]
    ],
)

HYPERELASTIC_MODELS = _models(
    hyperelastic,
    [
        ("ArrudaBoyce", {**_BS, "number_of_links": 8}, 3),
        ("BlatzKo", {**_BS, "mixing_parameter": 0.35}, 3),
        (
            "Carroll",
            {
                "bulk_modulus": 13,
                "linear_modulus": 1.2,
                "quartic_modulus": 1e-4,
                "second_invariant_modulus": 0.9,
            },
            None,
        ),
        ("Fung", {**_BS, "extra_modulus": 1, "exponent": 1}, 3),
        ("Gent", {**_BS, "extensibility": 23}, 3),
        ("Hencky", _BS, 3),
        ("Isihara", {**_BS, "extra_modulus": 1.2, "quadratic_modulus": 1e-2}, 3),
        ("MooneyRivlin", {**_BS, "extra_modulus": 1}, 3),
        ("NeoHookean", _BS, 3),
        (
            "Ogden",
            {
                "bulk_modulus": 13,
                "shear_moduli": [2.3, 1.1e-2, -2e-3],
                "exponents": [1.3, 5.0, -2.0],
            },
            None,
        ),
        ("SaintVenantKirchhoff", _BS, 3),
        (
            "Yeoh",
            {"bulk_modulus": 13, "shear_moduli": [3, -1, 3e-1, -1e-3, 1e-5]},
            3,
        ),
    ],
)


def _convert(key, value):
    if key.endswith(("modulus", "moduli")):
        return Stress.pascals(np.asarray(value, dtype=float))
    return value


class InPascals:
    """A model that takes stresses in pascals and returns them in pascals."""

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        attribute = getattr(self._model, name)
        if callable(attribute):
            return lambda *args: self._unwrap(attribute(*args))
        return self._unwrap(attribute)

    def __str__(self):
        return str(self._model)

    @staticmethod
    def _unwrap(value):
        return value.in_pascals() if isinstance(value, Stress) else value


def build(cls, params):
    return InPascals(cls(*[_convert(key, value) for key, value in params.items()]))


def assert_str(model, name, params):
    text = str(model)
    assert text.startswith(f"{name}(") and text.endswith(")")
    for key, value in params.items():
        if isinstance(value, list):
            assert f"{key}=[" in text
        else:
            assert f"{key}={float(value)}" in text


def assert_helmholtz_free_energy_density_zero(model, identity):
    assert model.helmholtz_free_energy_density(identity) == 0


def assert_first_piola_kirchhoff_stress_finite_difference(
    model, deformation_gradient, epsilon
):
    stress = model.first_piola_kirchhoff_stress(deformation_gradient)
    for i in range(3):
        for j in range(3):
            deformation_gradient[i, j] += epsilon / 2
            d_helmholtz = model.helmholtz_free_energy_density(deformation_gradient)
            deformation_gradient[i, j] -= epsilon
            d_helmholtz -= model.helmholtz_free_energy_density(deformation_gradient)
            assert np.abs(stress[i, j] - d_helmholtz / epsilon) < epsilon
            deformation_gradient[i, j] += epsilon / 2


def assert_first_piola_kirchhoff_tangent_stiffness_symmetry(
    model, deformation_gradient, abs_tol
):
    tan = model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    assert np.abs(tan[i, j, k, m] - tan[k, m, i, j]) < abs_tol


def assert_cauchy_stress_zero(model, identity, zero):
    assert (model.cauchy_stress(identity) == zero).all()


def assert_first_piola_kirchhoff_stress_zero(model, identity, zero):
    assert (model.first_piola_kirchhoff_stress(identity) == zero).all()


def assert_second_piola_kirchhoff_stress_zero(model, identity, zero):
    assert (model.second_piola_kirchhoff_stress(identity) == zero).all()


def assert_cauchy_stress_symmetry(model, deformation_gradient, abs_tol):
    assert (
        np.abs(
            model.cauchy_stress(deformation_gradient)
            - model.cauchy_stress(deformation_gradient).T
        )
        < abs_tol
    ).all()


def assert_cauchy_stress_relate_first_piola_kirchhoff_stress(
    model, deformation_gradient, abs_tol
):
    assert (
        model.cauchy_stress(deformation_gradient)
        - model.first_piola_kirchhoff_stress(deformation_gradient).dot(
            deformation_gradient.T
        )
        / np.linalg.det(deformation_gradient)
        < abs_tol
    ).all()


def assert_cauchy_stress_relate_second_piola_kirchhoff_stress(
    model, deformation_gradient, abs_tol
):
    assert (
        model.cauchy_stress(deformation_gradient)
        - deformation_gradient.dot(
            model.second_piola_kirchhoff_stress(deformation_gradient)
        ).dot(deformation_gradient.T)
        / np.linalg.det(deformation_gradient)
        < abs_tol
    ).all()


def assert_shear_modulus(model, simple_shear_small, epsilon, shear_modulus):
    assert (
        np.abs(model.cauchy_stress(simple_shear_small)[0, 1] / epsilon - shear_modulus)
        < epsilon
    )


def assert_bulk_modulus(model, volumetric_small, epsilon, bulk_modulus):
    assert (
        np.abs(
            model.cauchy_stress(volumetric_small).trace() / 3 / epsilon / bulk_modulus
            - 1
        )
        < 3 * epsilon
    )


def assert_cauchy_tangent_stiffness_finite_difference(
    model, deformation_gradient, abs_tol, epsilon
):
    tan = model.cauchy_tangent_stiffness(deformation_gradient)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    assert np.abs(tan[i, j, k, m] - tan[j, i, k, m]) < abs_tol
                    deformation_gradient[k, m] += epsilon / 2
                    d_stress = model.cauchy_stress(deformation_gradient)[i, j]
                    deformation_gradient[k, m] -= epsilon
                    d_stress -= model.cauchy_stress(deformation_gradient)[i, j]
                    assert np.abs(tan[i, j, k, m] - d_stress / epsilon) < 1.33 * epsilon
                    deformation_gradient[k, m] += epsilon / 2


def assert_first_piola_kirchhoff_tangent_stiffness_finite_difference(
    model, deformation_gradient, epsilon
):
    tan = model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    deformation_gradient[k, m] += epsilon / 2
                    d_stress = model.first_piola_kirchhoff_stress(deformation_gradient)[
                        i, j
                    ]
                    deformation_gradient[k, m] -= epsilon
                    d_stress -= model.first_piola_kirchhoff_stress(
                        deformation_gradient
                    )[i, j]
                    assert np.abs(tan[i, j, k, m] - d_stress / epsilon) < epsilon
                    deformation_gradient[k, m] += epsilon / 2


def assert_second_piola_kirchhoff_tangent_stiffness_finite_difference(
    model, deformation_gradient, epsilon
):
    tan = model.second_piola_kirchhoff_tangent_stiffness(deformation_gradient)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    deformation_gradient[k, m] += epsilon / 2
                    d_stress = model.second_piola_kirchhoff_stress(
                        deformation_gradient
                    )[i, j]
                    deformation_gradient[k, m] -= epsilon
                    d_stress -= model.second_piola_kirchhoff_stress(
                        deformation_gradient
                    )[i, j]
                    assert np.abs(tan[i, j, k, m] - d_stress / epsilon) < 2.33 * epsilon
                    deformation_gradient[k, m] += epsilon / 2
