from conspire.constitutive.solid.elastic import AlmansiHamel
import numpy as np


abs_tol = 1e-12
epsilon = 1e-6
bulk_modulus = 13
shear_modulus = 3
zero = np.zeros((3, 3))
identity = np.eye(3)
deformation_gradient = np.array([
    [0.63595746, 0.69157849, 0.71520784],
    [0.80589604, 0.83687323, 0.19312595],
    [0.05387420, 0.86551549, 0.41880244],
])
simple_shear_small = np.array([
    [1, epsilon, 0],
    [0, 1, 0],
    [0, 0, 1]
])
volumetric_small = identity * (1 + epsilon)**(1/3)

model = AlmansiHamel(bulk_modulus, shear_modulus)


def test_cauchy_stress_zero():
    assert (
        model.cauchy_stress(identity) == zero
    ).all()


def test_first_piola_kirchhoff_stress_zero():
    assert (
        model.first_piola_kirchhoff_stress(identity) == zero
    ).all()


def test_second_piola_kirchhoff_stress_zero():
    assert (
        model.second_piola_kirchhoff_stress(identity) == zero
    ).all()


def test_cauchy_stress_symmetry():
    assert (np.abs(
        model.cauchy_stress(deformation_gradient) -
        model.cauchy_stress(deformation_gradient).T
    ) < abs_tol).all()


def test_cauchy_stress_relate_first_piola_kirchhoff_stress():
    assert (
        model.cauchy_stress(deformation_gradient) -
        model.first_piola_kirchhoff_stress(deformation_gradient)
        .dot(deformation_gradient.T)
        / np.linalg.det(deformation_gradient)
        < abs_tol
    ).all()


def test_cauchy_stress_relate_second_piola_kirchhoff_stress():
    assert (
        model.cauchy_stress(deformation_gradient) -
        deformation_gradient
        .dot(model.second_piola_kirchhoff_stress(deformation_gradient))
        .dot(deformation_gradient.T)
        / np.linalg.det(deformation_gradient)
        < abs_tol
    ).all()


def test_shear_modulus():
    assert np.abs(
        model.cauchy_stress(simple_shear_small)[0, 1]
        / epsilon - shear_modulus
    ) < epsilon


def test_bulk_modulus():
    assert np.abs(
        model.cauchy_stress(volumetric_small).trace()
        / 3 / epsilon / bulk_modulus - 1
    ) < 3 * epsilon


def test_cauchy_tangent_stiffness_finite_difference():
    tan = model.cauchy_tangent_stiffness(deformation_gradient)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    assert np.abs(
                        tan[i, j, k, m] - tan[j, i, k, m]
                    ) < abs_tol
                    deformation_gradient[k, m] += epsilon / 2
                    d_stress = model.cauchy_stress(
                        deformation_gradient
                    )[i, j]
                    deformation_gradient[k, m] -= epsilon
                    d_stress -= model.cauchy_stress(
                        deformation_gradient
                    )[i, j]
                    assert np.abs(
                        tan[i, j, k, m] - d_stress / epsilon
                    ) < 1.33 * epsilon
                    deformation_gradient[k, m] += epsilon / 2


def test_first_piola_kirchhoff_tangent_stiffness_finite_difference():
    tan = model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    deformation_gradient[k, m] += epsilon / 2
                    d_stress = model.first_piola_kirchhoff_stress(
                        deformation_gradient
                    )[i, j]
                    deformation_gradient[k, m] -= epsilon
                    d_stress -= model.first_piola_kirchhoff_stress(
                        deformation_gradient
                    )[i, j]
                    assert np.abs(
                        tan[i, j, k, m] - d_stress / epsilon
                    ) < epsilon
                    deformation_gradient[k, m] += epsilon / 2


def test_second_piola_kirchhoff_tangent_stiffness_finite_difference():
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
                    assert np.abs(
                        tan[i, j, k, m] - d_stress / epsilon
                    ) < 2.33 * epsilon
                    deformation_gradient[k, m] += epsilon / 2
