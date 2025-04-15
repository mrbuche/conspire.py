from conspire.constitutive.solid.hyperelastic import NeoHookean
from conspire.fem import Block
import numpy as np


abs_tol = 1e-12
epsilon = 1e-6
bulk_modulus = 13
shear_modulus = 3
deformation_gradient = np.array([
    [0.63595746, 0.69157849, 0.71520784],
    [0.80589604, 0.83687323, 0.19312595],
    [0.05387420, 0.86551549, 0.41880244],
])
connectivity = np.array([
    [13, 12, 8, 1],
    [10, 3, 0, 8],
    [11, 10, 8, 3],
    [12, 11, 8, 2],
    [11, 2, 3, 8],
    [12, 2, 8, 1],
    [13, 10, 5, 0],
    [13, 11, 10, 8],
    [10, 6, 9, 5],
    [12, 7, 4, 9],
    [12, 11, 7, 9],
    [11, 7, 9, 6],
    [13, 1, 8, 0],
    [13, 9, 4, 5],
    [13, 12, 1, 4],
    [11, 10, 6, 9],
    [11, 10, 3, 6],
    [12, 11, 2, 7],
    [13, 11, 9, 10],
    [13, 12, 4, 9],
    [13, 10, 0, 8],
    [13, 10, 9, 5],
    [13, 12, 11, 8],
    [13, 12, 9, 11],
])
reference_coordinates = np.array([
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.0, 0.0, 0.5],
    [0.0, 0.0, -0.5],
    [0.0, -0.5, 0.0],
    [-0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.5, 0.0, 0.0],
])
deformed_coordinates = np.array([
    [0.48419081, -0.52698494, 0.42026988],
    [0.43559430, 0.52696224, 0.54477963],
    [-0.56594965, 0.57076191, 0.51683869],
    [-0.56061746, -0.42795457, 0.55275658],
    [0.41878700, 0.53190268, -0.44744274],
    [0.47232357, -0.57252738, -0.42946606],
    [-0.45168197, -0.5102938, -0.57959825],
    [-0.41776733, 0.41581785, -0.45911886],
    [0.05946988, 0.03773822, 0.44149305],
    [-0.08478334, -0.09009810, -0.46105872],
    [-0.04039882, -0.58201398, 0.09346960],
    [-0.57820738, 0.08325131, 0.03614415],
    [-0.04145077, 0.56406301, 0.09988905],
    [0.52149656, -0.08553510, -0.03187069],
])
affinely_deformed_coordinates = np.zeros(reference_coordinates.shape)
for i, reference_coordinate in enumerate(reference_coordinates):
    affinely_deformed_coordinates[i] = \
        deformation_gradient.dot(reference_coordinate)

model = NeoHookean(bulk_modulus, shear_modulus)

block = Block(model, connectivity, reference_coordinates)


def test_helmholtz_free_energy_zero():
    assert block.helmholtz_free_energy(reference_coordinates) == 0


def test_helmholtz_free_energy_affine():
    assert np.abs(
        block.helmholtz_free_energy(affinely_deformed_coordinates) -
        model.helmholtz_free_energy_density(deformation_gradient)
    ) < abs_tol


def test_nodal_forces_zero():
    assert np.all(block.nodal_forces(reference_coordinates) == 0.0)


def test_nodal_forces_finite_difference_undeformed():
    for a in range(len(reference_coordinates)):
        for i in range(3):
            reference_coordinates[a, i] += epsilon / 2
            d_helmholtz = block.helmholtz_free_energy(
                reference_coordinates
            )
            reference_coordinates[a, i] -= epsilon
            d_helmholtz -= block.helmholtz_free_energy(
                reference_coordinates
            )
            assert np.abs(d_helmholtz / epsilon) < epsilon
            reference_coordinates[a, i] += epsilon / 2


def test_nodal_forces_finite_difference_deformed():
    forces = block.nodal_forces(deformed_coordinates)
    for a in range(len(reference_coordinates)):
        for i in range(3):
            deformed_coordinates[a, i] += epsilon / 2
            d_helmholtz = block.helmholtz_free_energy(
                deformed_coordinates
            )
            deformed_coordinates[a, i] -= epsilon
            d_helmholtz -= block.helmholtz_free_energy(
                deformed_coordinates
            )
            assert np.abs(
                forces[a, i] - d_helmholtz / epsilon
            ) < epsilon
            deformed_coordinates[a, i] += epsilon / 2
