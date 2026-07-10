import numpy as np


def assert_str(model, name, extra, bulk_modulus, shear_modulus):
    expected = f"{name}(bulk_modulus={bulk_modulus}, shear_modulus={shear_modulus}"
    for key, value in extra.items():
        expected += f", {key}={value}"
    expected += ")"
    assert model.__str__() == expected


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
